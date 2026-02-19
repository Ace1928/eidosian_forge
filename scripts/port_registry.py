#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT):
    s = str(extra)
    if extra.exists() and s not in sys.path:
        sys.path.insert(0, s)

from eidosian_core import (
    detect_registry_path,
    get_service_config,
    get_service_host,
    get_service_port,
    get_service_url,
    load_port_registry,
)


def cmd_get(args: argparse.Namespace) -> int:
    if args.field == "port":
        default_port = int(args.default) if str(args.default).strip() else 0
        value = get_service_port(
            args.service,
            default=default_port,
            env_keys=args.env if args.env else None,
            registry_path=args.registry,
        )
        print(value)
        return 0

    if args.field == "host":
        value = get_service_host(
            args.service,
            default=args.default,
            env_keys=args.env if args.env else None,
            registry_path=args.registry,
        )
        print(value)
        return 0

    if args.field == "path":
        config = get_service_config(args.service, args.registry)
        print(str(config.get("path", args.default)))
        return 0

    config = get_service_config(args.service, args.registry)
    value = config.get(args.field, args.default)
    if isinstance(value, (dict, list)):
        print(json.dumps(value))
    else:
        print(value)
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    url = get_service_url(
        args.service,
        default_port=int(args.default_port),
        default_host=args.default_host,
        default_protocol=args.default_protocol,
        default_path=args.default_path,
        registry_path=args.registry,
    )
    print(url)
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    registry_path = args.registry if args.registry else str(detect_registry_path())
    registry = load_port_registry(registry_path)
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and query Eidosian Forge port registry.")
    parser.add_argument("--registry", default="", help="Override registry path (default: auto-detect config/ports.json)")

    sub = parser.add_subparsers(dest="command", required=True)

    get_parser = sub.add_parser("get", help="Get a field for a service")
    get_parser.add_argument("--service", required=True)
    get_parser.add_argument("--field", default="port", help="Field to retrieve (port|host|path|any json key)")
    get_parser.add_argument("--default", default="", help="Fallback if field not found")
    get_parser.add_argument("--env", action="append", help="Explicit env key(s) to check before registry")
    get_parser.set_defaults(func=cmd_get)

    url_parser = sub.add_parser("url", help="Build service URL")
    url_parser.add_argument("--service", required=True)
    url_parser.add_argument("--default-port", type=int, default=0)
    url_parser.add_argument("--default-host", default="127.0.0.1")
    url_parser.add_argument("--default-protocol", default="http")
    url_parser.add_argument("--default-path", default="")
    url_parser.set_defaults(func=cmd_url)

    dump_parser = sub.add_parser("dump", help="Dump full registry JSON")
    dump_parser.set_defaults(func=cmd_dump)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
