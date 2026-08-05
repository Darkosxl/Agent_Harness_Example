# Quickstart

```bash
# setup
uv venv .venv
uv pip install --python .venv/bin/python arc-agi arcengine openai pygame
echo 'CEREBRAS_API_KEY=your-key-here' > .env

# sanity checks
.venv/bin/python arc3_harness.py --selftest
.venv/bin/python arc3_harness.py --list

# play easiest game until finish (watch pygame window + reasoning in terminal)
.venv/bin/python arc3_harness.py --game ft09 --max-steps 100000

# play one game, capped steps
.venv/bin/python arc3_harness.py --game ls20 --max-steps 100

# all 25 public games
.venv/bin/python arc3_harness.py

# options
.venv/bin/python arc3_harness.py --game vc33 --model gemma-4-31b --ctx-tokens 60000 --max-reply 800 --headless
```
