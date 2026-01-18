# Code Forge Narrative Engine

This repository contains a proof-of-concept narrative engine that maintains
persistent memory of conversations and events. The system can run in an
interactive chat mode from the command line, storing all interactions in a
local JSON file. During idle periods the engine generates autonomous
"thoughts" that are also recorded. If the configured model cannot be loaded the
engine now stops with a clear error instead of falling back to an echo mode.

## Features

- **Persistent memory** stored in `memory.json` by default.
- **Glossary** that counts words the engine encounters.
- **Idle thinking**: the engine reflects on the last message when no input has
  been received for a configurable period.
- **Command line interface** with subcommands to chat and inspect memory.
- **Model-based responses** powered by a small open-source language model.

## Installation

Run the bundled `install.sh` script to create a virtual environment and install
dependencies in editable mode:

```bash
./install.sh
```

Activate the environment with `source .venv/bin/activate` and you are ready to
run the `forgengine` command.

## Usage

Run the chat interface. By default the engine uses the
`Qwen/Qwen3-1.7B-FP8`
model from HuggingFace:

```bash
forgengine chat
```

To select a different model:

```bash
forgengine --model sshleifer/tiny-gpt2 chat
```

Use `--max-tokens` to control the length of responses:

```bash
forgengine --max-tokens 128 chat
```

View stored interactions:

```bash
forgengine memory
```

Other subcommands include `events` and `glossary`. Use `--help` for details.

### Interactive setup

Running `forgengine` with no existing configuration walks you through a short
setup. It scans for local HuggingFace models and lets you pick one, storing your
choice along with other options in `~/.forgengine.json`. Use `--setup` at any
time to reconfigure. A `models` subcommand lists discovered models.

Optionally specify an adapter path with `--adapter` to apply LoRA weights if the
`peft` package is available.

If the selected model cannot be loaded an error is printed and the program
terminates so you always know when something is wrong.

