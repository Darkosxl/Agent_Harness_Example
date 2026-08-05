#!/usr/bin/env python3
"""Minimal ARC-AGI-3 agentic harness: one LLM looping on itself.

Memory = raw message list only. No compaction, no summaries — oldest
turns are hard-dropped once the history passes --ctx-tokens. Pygame
window shows the game live; the model's reasoning streams to stdout.

Usage:
    .venv/bin/python arc3_harness.py --selftest
    .venv/bin/python arc3_harness.py --list
    .venv/bin/python arc3_harness.py --game ls20 --max-steps 50
    .venv/bin/python arc3_harness.py              # all public games

Needs CEREBRAS_API_KEY exported for live play.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

MOVE_RE = re.compile(r"MOVE:\s*(RESET|ACTION[1-7])(?:\s+(\d+)[\s,]+(\d+))?", re.I)

PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (85, 85, 85), (255, 255, 255),
    (0, 255, 255), (128, 0, 128), (0, 128, 0), (128, 128, 0),
]

SYSTEM = """You are playing a game. Find out the goal of the game, record what changes in your reasoning, figure out the important objects of the game.

You see a 64x64 grid of hex digits (0-f); each digit is a colored cell.
Nobody tells you the rules. Experiment, watch what changes between turns, form and test theories, complete levels to raise your level count.
Moves available in this game: {moves}.
Coordinates: x = column 0-63 (left to right), y = row 0-63 (top to bottom).
RESET restarts the game — use it after GAME_OVER or when hopelessly stuck.
Think out loud briefly (what changed? what does that suggest? what to try next?), then end your reply with exactly one line:
MOVE: <name>
or, for coordinate moves:
MOVE: <name> <x> <y>"""


def grid_text(grid):
    return "\n".join("".join(format(v, "x") for v in row) for row in grid)


def parse_move(text):
    hits = MOVE_RE.findall(text or "")
    if not hits:
        return None
    name, x, y = hits[-1]
    return name.upper(), ((int(x), int(y)) if x and y else None)


def truncate(messages, budget_tokens):
    # ponytail: chars//4 ~ tokens; drop oldest user+assistant pair, keep system
    def est():
        return sum(len(m["content"]) for m in messages) // 4
    while est() > budget_tokens and len(messages) > 3:
        del messages[1:3]


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

    def pump(self):  # no-op; thread pumps continuously
        pass

    def show(self, frames, caption):
        with self.lock:
            self.latest = (frames, caption)


def make_client(model):
    import openai
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        sys.exit("CEREBRAS_API_KEY not set")
    client = openai.OpenAI(base_url="https://api.cerebras.ai/v1", api_key=key)
    try:
        ids = [m.id for m in client.models.list()]
        if model not in ids:
            sys.exit(f"Model {model!r} not on Cerebras. Available: {ids}")
    except openai.OpenAIError:
        pass  # listing unsupported; first chat call will surface real errors
    return client


def stream_reply(client, model, messages, viewer, max_tokens):
    out = []
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, max_tokens=max_tokens)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            out.append(delta)
            print(delta, end="", flush=True)
        if viewer:
            viewer.pump()
    print()
    return "".join(out)


def play_game(arc, gid, client, args, viewer):
    from arcengine import GameState

    env = arc.make(gid)
    if env is None:
        print(f"could not create env for {gid!r}, skipping")
        return None
    obs = env.reset()
    space = {a.name: a for a in env.action_space}
    moves = ", ".join(a.name + (" (needs x y)" if a.is_complex() else "")
                      for a in env.action_space)
    messages = [{"role": "system", "content": SYSTEM.format(moves=moves)}]
    prev, note, steps = None, "", 0

    for step in range(1, args.max_steps + 1):
        grid = obs.frame[-1]
        changed = "?" if prev is None else sum(
            a != b for ra, rb in zip(prev, grid) for a, b in zip(ra, rb))
        messages.append({"role": "user", "content": (
            f"{note}Step {step} | state={obs.state.name} | "
            f"levels {obs.levels_completed}/{obs.win_levels} | "
            f"cells changed since last frame: {changed}\n{grid_text(grid)}")})
        note = ""
        truncate(messages, args.ctx_tokens)

        print(f"\n--- {gid} step {step} | {obs.state.name} | "
              f"lvl {obs.levels_completed}/{obs.win_levels} ---")
        reply = stream_reply(client, args.model, messages, viewer, args.max_reply)
        messages.append({"role": "assistant", "content": reply})

        parsed = parse_move(reply)
        if parsed and (parsed[0] == "RESET" or parsed[0] in space):
            name, coords = parsed
        else:
            name, coords = random.choice(list(space)), None
            note = f"(your last reply had no valid MOVE line; harness played {name} at random)\n"

        prev, steps, data = grid, step, {}
        if name == "RESET":
            obs2 = env.reset()
        else:
            action = space[name]
            if action.is_complex():
                if coords is None:
                    coords = (random.randint(0, 63), random.randint(0, 63))
                    note += "(no coords given; harness picked random x y)\n"
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


def selftest():
    assert parse_move("thinking...\nMOVE: ACTION3") == ("ACTION3", None)
    assert parse_move("MOVE: action6 10 20") == ("ACTION6", (10, 20))
    assert parse_move("MOVE: ACTION1\nwait, no.\nMOVE: RESET") == ("RESET", None)
    assert parse_move("no move here") is None
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 400}] * 10
    truncate(msgs, 500)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and len(msgs) == 5
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", help="comma-separated short game ids, e.g. ls20,vc33")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--ctx-tokens", type=int, default=60_000,
                   help="history budget; older turns beyond it are dropped")
    p.add_argument("--max-reply", type=int, default=800)
    p.add_argument("--model", default=None)
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
    args.model = args.model or os.environ.get("CEREBRAS_MODEL", "gemma-4-31b")

    if args.selftest:
        return selftest()

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
    client = make_client(args.model)

    results = []
    for i, gid in enumerate(game_ids, 1):
        print(f"\n===== [{i}/{len(game_ids)}] {gid} =====")
        r = play_game(arc, gid, client, args, viewer)
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
