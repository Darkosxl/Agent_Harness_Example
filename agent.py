"""The agent: Cerebras LLM looping on a raw message list.

Memory = the message list only. No compaction — oldest user/assistant
pairs are hard-dropped once past the token budget. Reasoning streams
to stdout as it arrives.
"""
from __future__ import annotations

import os
import re
import sys

MOVE_RE = re.compile(r"MOVE:\s*(RESET|ACTION[1-7])(?:\s+(\d+)[\s,]+(\d+))?", re.I)

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


PROVIDERS = {
    "cerebras": dict(base_url="https://api.cerebras.ai/v1",
                     key_env="CEREBRAS_API_KEY", model_env="CEREBRAS_MODEL",
                     default_model="gemma-4-31b"),
    "deepinfra": dict(base_url="https://api.deepinfra.com/v1/openai",
                      key_env="DEEPINFRA_API_KEY", model_env="DEEPINFRA_MODEL",
                      default_model="Qwen/Qwen3.6-27B"),
}


class Agent:
    def __init__(self, model=None, ctx_tokens=60_000, max_reply=800,
                 provider="cerebras"):
        import openai
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            sys.exit(f"{cfg['key_env']} not set")
        model = model or os.environ.get(cfg["model_env"]) or cfg["default_model"]
        self.client = openai.OpenAI(base_url=cfg["base_url"], api_key=key)
        try:
            ids = [m.id for m in self.client.models.list()]
            if model not in ids:
                import difflib
                close = ([i for i in ids if model.lower().split("/")[-1] in i.lower()]
                         or difflib.get_close_matches(model, ids, n=10, cutoff=0.3)
                         or ids[:20])
                sys.exit(f"Model {model!r} not on {provider}. Close matches: {close}")
        except openai.OpenAIError:
            pass  # listing unsupported; first chat call will surface real errors
        print(f"agent: {provider} / {model}")
        self.model = model
        self.ctx_tokens = ctx_tokens
        self.max_reply = max_reply
        self.messages = []
        self.pending_note = ""

    def start_game(self, moves_desc):
        """Fresh memory for a new game."""
        self.messages = [
            {"role": "system", "content": SYSTEM.format(moves=moves_desc)}]
        self.pending_note = ""

    def add_note(self, text):
        """Harness feedback shown to the model at the start of its next turn."""
        self.pending_note += text

    def choose(self, observation):
        """One reasoning loop: observe -> stream reasoning -> parsed move.

        Returns (reply_text, parsed) where parsed is (NAME, (x, y) | None)
        or None if the reply had no valid MOVE line.
        """
        self.messages.append(
            {"role": "user", "content": self.pending_note + observation})
        self.pending_note = ""
        truncate(self.messages, self.ctx_tokens)
        reply = self._stream()
        self.messages.append({"role": "assistant", "content": reply})
        return reply, parse_move(reply)

    def _stream(self):
        out = []
        stream = self.client.chat.completions.create(
            model=self.model, messages=self.messages,
            stream=True, max_tokens=self.max_reply)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                out.append(delta)
                print(delta, end="", flush=True)
        print()
        return "".join(out)

    @classmethod
    def selftest(cls):
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
