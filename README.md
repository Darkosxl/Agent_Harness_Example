# Quickstart

```bash
# setup
uv venv .venv
uv pip install --python .venv/bin/python arc-agi arcengine openai pygame
# keys: set the one(s) for the provider you use
echo 'CEREBRAS_API_KEY=your-key-here' > .env
echo 'DEEPINFRA_API_KEY=your-key-here' >> .env
# optional defaults: PROVIDER=cerebras|deepinfra, CEREBRAS_MODEL=..., DEEPINFRA_MODEL=...

# sanity checks
.venv/bin/python arc3_harness.py --selftest
.venv/bin/python arc3_harness.py --list

# play easiest game until finish (watch pygame window + reasoning in terminal)
.venv/bin/python arc3_harness.py --game ft09 --max-steps 100000

# play one game, capped steps
.venv/bin/python arc3_harness.py --game ls20 --max-steps 100

# all 25 public games
.venv/bin/python arc3_harness.py

# providers
.venv/bin/python arc3_harness.py --game ft09 --provider cerebras   # gemma-4-31b (default)
.venv/bin/python arc3_harness.py --game ft09 --provider deepinfra  # Qwen/Qwen3.6-27B

# options
.venv/bin/python arc3_harness.py --game vc33 --provider deepinfra --model Qwen/Qwen3.6-27B --ctx-tokens 60000 --max-reply 800 --headless

.venv/bin/python arc3_harness.py --game ft09                        # cerebras + gemma
.venv/bin/python arc3_harness.py --game ft09 --provider deepinfra   # qwen

```

