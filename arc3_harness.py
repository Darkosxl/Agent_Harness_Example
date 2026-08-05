#!/usr/bin/env python3
"""Minimal ARC-AGI-3 harness: game engine + pygame viewer + main loop.

The LLM lives in agent.py (class Agent); this file only drives it.

Usage:
    .venv/bin/python arc3_harness.py --selftest
    .venv/bin/python arc3_harness.py --list
    .venv/bin/python arc3_harness.py --game ft09 --max-steps 100000
    .venv/bin/python arc3_harness.py              # all public games

Needs CEREBRAS_API_KEY exported (or in .env) for live play.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

from agent import Agent

PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (85, 85, 85), (255, 255, 255),
    (0, 255, 255), (128, 0, 128), (0, 128, 0), (128, 128, 0),
]


def grid_text(grid):
    return "\n".join("".join(format(v, "x") for v in row) for row in grid)


class Viewer:
    CELL = 10

    def __init__(self):
        if os.environ.get("WAYLAND_DISPLAY"):
            # XWayland GLX is broken here; use native wayland driver
            os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
        import threading
        self.latest = None  # (frames, caption)
        self.alive = True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        # ponytail: pygame owns its thread so the window keeps pumping
        # events while the main loop blocks on the LLM or the game engine
        import pygame
        pygame.init()
        side = 64 * self.CELL
        # resizable + scale-blit: tiling WMs force arbitrary window sizes;
        # a fixed-size surface there renders stride garbage
        screen = pygame.display.set_mode((side, side), pygame.RESIZABLE)
        canvas = pygame.Surface((side, side))
        clock = pygame.time.Clock()
        while self.alive:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.alive = False
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
            with self.lock:
                job, self.latest = self.latest, None
            if job:
                frames, caption = job
                pygame.display.set_caption(caption)
                for grid in frames:
                    for y, row in enumerate(grid):
                        for x, v in enumerate(row):
                            canvas.fill(
                                PALETTE[v & 15],
                                (x * self.CELL, y * self.CELL, self.CELL, self.CELL))
                    if len(frames) > 1:
                        pygame.transform.scale(canvas, screen.get_size(), screen)
                        pygame.display.flip()
                        pygame.time.wait(60)
            pygame.transform.scale(canvas, screen.get_size(), screen)
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def show(self, frames, caption):
        with self.lock:
            self.latest = (frames, caption)


def play_game(arc, gid, agent, args, viewer):
    from arcengine import GameState

    env = arc.make(gid)
    if env is None:
        print(f"could not create env for {gid!r}, skipping")
        return None
    obs = env.reset()
    space = {a.name: a for a in env.action_space}
    moves = ", ".join(a.name + (" (needs x y)" if a.is_complex() else "")
                      for a in env.action_space)
    agent.start_game(moves)
    prev, steps = None, 0

    for step in range(1, args.max_steps + 1):
        grid = obs.frame[-1]
        changed = "?" if prev is None else sum(
            a != b for ra, rb in zip(prev, grid) for a, b in zip(ra, rb))
        observation = (
            f"Step {step} | state={obs.state.name} | "
            f"levels {obs.levels_completed}/{obs.win_levels} | "
            f"cells changed since last frame: {changed}\n{grid_text(grid)}")

        print(f"\n--- {gid} step {step} | {obs.state.name} | "
              f"lvl {obs.levels_completed}/{obs.win_levels} ---")
        reply, parsed = agent.choose(observation)

        if parsed and (parsed[0] == "RESET" or parsed[0] in space):
            name, coords = parsed
        else:
            name, coords = random.choice(list(space)), None
            agent.add_note(f"(your last reply had no valid MOVE line; "
                           f"harness played {name} at random)\n")

        prev, steps, data = grid, step, {}
        if name == "RESET":
            obs2 = env.reset()
        else:
            action = space[name]
            if action.is_complex():
                if coords is None:
                    coords = (random.randint(0, 63), random.randint(0, 63))
                    agent.add_note("(no coords given; harness picked random x y)\n")
                data = {"x": min(coords[0], 63), "y": min(coords[1], 63)}
            obs2 = env.step(action, data=data, reasoning={"text": reply[-2000:]})
        if obs2 is not None:
            obs = obs2

        print(f">> {name} {data or ''} -> {obs.state.name}, "
              f"lvl {obs.levels_completed}/{obs.win_levels}")
        if viewer:
            viewer.show(obs.frame, f"{gid} | lvl {obs.levels_completed}/"
                                   f"{obs.win_levels} | {obs.state.name} | step {step}")
        if obs.state == GameState.WIN:
            print(f"*** WIN {gid} in {step} steps ***")
            break

    return gid, obs.state.name, obs.levels_completed, steps


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", help="comma-separated short game ids, e.g. ls20,vc33")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--ctx-tokens", type=int, default=60_000,
                   help="history budget; older turns beyond it are dropped")
    p.add_argument("--max-reply", type=int, default=800)
    p.add_argument("--model", default=None,
                   help="provider model id; defaults per provider")
    p.add_argument("--provider", choices=["cerebras", "deepinfra"], default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    # ponytail: tiny .env loader, no dotenv dep
    try:
        for line in open(".env"):
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#") and v:
                os.environ.setdefault(k, v.strip().strip('"'))
    except FileNotFoundError:
        pass
    args.provider = args.provider or os.environ.get("PROVIDER", "cerebras")

    if args.selftest:
        return Agent.selftest()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    all_ids = sorted({e.game_id.split("-")[0] for e in envs})

    if args.list:
        print(f"{len(all_ids)} games:")
        for e in envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    if args.game:
        wanted = [g.strip() for g in args.game.split(",")]
        unknown = set(wanted) - set(all_ids)
        if unknown:
            sys.exit(f"unknown game id(s): {sorted(unknown)}; run --list")
        game_ids = wanted
    else:
        game_ids = all_ids
        print(f"playing all {len(game_ids)} games")

    viewer = None if args.headless else Viewer()
    agent = Agent(args.model, args.ctx_tokens, args.max_reply,
                  provider=args.provider)

    results = []
    for i, gid in enumerate(game_ids, 1):
        print(f"\n===== [{i}/{len(game_ids)}] {gid} =====")
        r = play_game(arc, gid, agent, args, viewer)
        if r:
            results.append(r)

    print("\n========= SUMMARY =========")
    for gid, state, levels, steps in results:
        print(f"  {gid:8} levels={levels:3}  steps={steps:5}  state={state}")
    sc = arc.get_scorecard()
    if sc is not None:
        print(f"aggregate score: {getattr(sc, 'score', sc)}")


if __name__ == "__main__":
    main()
