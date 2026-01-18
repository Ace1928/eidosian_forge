"""Command line interface for the Narrative Engine."""

import argparse
import json
import os
from typing import List

from .engine import NarrativeEngine, PRIMARY_MODEL
from .memory import MemoryStore


CONFIG_PATH = os.path.expanduser("~/.forgengine.json")


def discover_local_models(directories: List[str] | None = None) -> List[str]:
    """Return possible local model directories."""
    if directories is None:
        directories = [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/models"),
            os.path.join(os.getcwd(), "models"),
        ]
    found: List[str] = []
    for base in directories:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if "config.json" in files:
                found.append(root)
    return sorted(found)

def load_config(path: str = CONFIG_PATH, force_setup: bool = False) -> dict:
    """Load configuration or run interactive setup."""
    path = os.path.expanduser(path)
    if force_setup or not os.path.exists(path):
        print("Setting up ForgeEngine configuration.")
        memory = input("Memory file path [memory.json]: ") or "memory.json"
        think = input("Think interval seconds [10]: ")
        think = int(think) if think else 10

        discovered = discover_local_models()
        default_model = discovered[0] if discovered else PRIMARY_MODEL
        if discovered:
            print("Available local models:")
            for idx, name in enumerate(discovered, 1):
                print(f"  {idx}. {name}")
        model = input(f"Model name [{default_model}]: ") or default_model

        tokens = input("Max tokens [512]: ")
        tokens = int(tokens) if tokens else 512
        adapter = input("Adapter path [none]: ") or ""

        config = {
            "memory": memory,
            "think": think,
            "model": model,
            "max_tokens": tokens,
            "adapter": adapter,
        }
        save = input("Save configuration for next time? [Y/n]: ").strip().lower()
        if not save or save.startswith("y"):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2)
            print(f"Configuration saved to {path}")
        return config

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_chat(args: argparse.Namespace) -> None:
    try:
        engine = NarrativeEngine(
            memory_path=args.memory,
            think_interval=args.think,
            model_name=args.model,
            max_tokens=args.max_tokens,
            adapter_path=args.adapter,
        )
    except Exception as exc:
        print(f"Failed to start engine: {exc}")
        return
    engine._reset_timer()
    print("Type 'quit' or 'exit' to stop.")
    try:
        while True:
            user_input = input("> ")
            if user_input.lower().strip() in {"quit", "exit"}:
                break
            print(engine.respond(user_input))
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()


def show_memory(args: argparse.Namespace) -> None:
    store = MemoryStore(args.memory)
    for item in store.data.interactions:
        print(f"{item['timestamp']}: {item['user']} -> {item['response']}")


def show_events(args: argparse.Namespace) -> None:
    store = MemoryStore(args.memory)
    for evt in store.data.events:
        print(f"{evt['timestamp']}: {evt['event']}")


def show_glossary(args: argparse.Namespace) -> None:
    store = MemoryStore(args.memory)
    for word, count in sorted(store.data.glossary.items()):
        print(f"{word}: {count}")

def list_models(args: argparse.Namespace) -> None:
    for path in discover_local_models():
        print(path)

def build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrative Engine CLI")
    parser.add_argument("--memory", default=config.get("memory", "memory.json"), help="Memory file path")
    parser.add_argument(
        "--think", type=int, default=config.get("think", 10), help="Seconds of idle before thinking"
    )
    parser.add_argument(
        "--model",
        default=config.get("model", PRIMARY_MODEL),
        help="HuggingFace model name to use for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.get("max_tokens", 512),
        help="Maximum tokens to generate for each response",
    )
    parser.add_argument(
        "--adapter",
        default=config.get("adapter", ""),
        help="Optional adapter or LoRA weights path",
    )

    sub = parser.add_subparsers(dest="command")
    chat = sub.add_parser("chat", help="Start interactive chat")
    chat.set_defaults(func=run_chat)

    mem = sub.add_parser("memory", help="Show stored interactions")
    mem.set_defaults(func=show_memory)

    evt = sub.add_parser("events", help="Show recorded events")
    evt.set_defaults(func=show_events)

    glo = sub.add_parser("glossary", help="Show learned glossary")
    glo.set_defaults(func=show_glossary)

    lm = sub.add_parser("models", help="List available local models")
    lm.set_defaults(func=list_models)

    return parser


def main() -> None:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=CONFIG_PATH)
    base.add_argument("--setup", action="store_true")
    known, remaining = base.parse_known_args()

    config = load_config(known.config, force_setup=known.setup)

    parser = build_parser(config)
    parser.add_argument("--config", default=known.config, help=argparse.SUPPRESS)
    parser.add_argument("--setup", action="store_true", help="Run interactive setup")

    args = parser.parse_args(remaining)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

