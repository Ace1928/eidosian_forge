#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_alerts(repo: str, state: str) -> list[dict[str, Any]]:
    cmd = [
        "gh",
        "api",
        "-X",
        "GET",
        f"repos/{repo}/dependabot/alerts",
        "--paginate",
    ]
    if state:
        cmd.extend(["-F", f"state={state}"])
    try:
        out = subprocess.check_output(cmd, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required but not installed") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise RuntimeError(
            f"gh api call failed (exit={exc.returncode}): {stderr.strip()}"
        ) from exc

    items: list[dict[str, Any]] = []
    text = out.strip()
    if not text:
        return items

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # gh --paginate can return multiple JSON arrays split by lines.
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            part = json.loads(raw)
        except Exception:
            continue
        if isinstance(part, list):
            items.extend([x for x in part if isinstance(x, dict)])
        elif isinstance(part, dict):
            items.append(part)
    return items


def _summary(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    by_state = Counter()
    by_severity = Counter()
    by_ecosystem = Counter()
    by_package = Counter()
    by_manifest = Counter()

    high_critical: list[dict[str, Any]] = []
    for row in alerts:
        state = str(row.get("state") or "unknown")
        vuln = row.get("security_vulnerability") if isinstance(row.get("security_vulnerability"), dict) else {}
        dep = row.get("dependency") if isinstance(row.get("dependency"), dict) else {}
        package = dep.get("package") if isinstance(dep.get("package"), dict) else {}
        advisory = row.get("security_advisory") if isinstance(row.get("security_advisory"), dict) else {}

        severity = str(vuln.get("severity") or "unknown").lower()
        ecosystem = str(package.get("ecosystem") or "unknown")
        pkg_name = str(package.get("name") or "unknown")
        manifest = str(dep.get("manifest_path") or "unknown")

        by_state[state] += 1
        by_severity[severity] += 1
        by_ecosystem[ecosystem] += 1
        by_package[pkg_name] += 1
        by_manifest[manifest] += 1

        if state == "open" and severity in {"critical", "high"}:
            high_critical.append(
                {
                    "number": row.get("number"),
                    "severity": severity,
                    "ecosystem": ecosystem,
                    "package": pkg_name,
                    "manifest_path": manifest,
                    "summary": str(advisory.get("summary") or ""),
                }
            )

    return {
        "totals": {
            "alerts": len(alerts),
            "open": int(by_state.get("open", 0)),
            "fixed": int(by_state.get("fixed", 0)),
        },
        "by_state": dict(by_state),
        "by_severity": dict(by_severity),
        "open_by_severity": {
            key: sum(
                1
                for row in alerts
                if str(row.get("state") or "") == "open"
                and str(
                    (
                        row.get("security_vulnerability")
                        if isinstance(row.get("security_vulnerability"), dict)
                        else {}
                    ).get("severity")
                    or ""
                ).lower()
                == key
            )
            for key in sorted(by_severity.keys())
        },
        "by_ecosystem": dict(by_ecosystem),
        "top_packages": by_package.most_common(25),
        "top_manifest_paths": by_manifest.most_common(25),
        "open_high_critical": high_critical,
    }


def _print_human(summary: dict[str, Any]) -> None:
    totals = summary.get("totals") or {}
    print(f"alerts={totals.get('alerts', 0)} open={totals.get('open', 0)} fixed={totals.get('fixed', 0)}")
    print("open_by_severity:")
    for key, value in (summary.get("open_by_severity") or {}).items():
        print(f"  {key}: {value}")
    print("top_manifest_paths:")
    for path, count in (summary.get("top_manifest_paths") or [])[:10]:
        print(f"  {count:>4}  {path}")
    print("top_packages:")
    for pkg, count in (summary.get("top_packages") or [])[:12]:
        print(f"  {count:>4}  {pkg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Dependabot alerts for a GitHub repository")
    parser.add_argument("--repo", default="Ace1928/eidosian_forge", help="owner/repo")
    parser.add_argument("--state", default="open", choices=["open", "fixed", "dismissed", "auto_dismissed", ""], help="filter state")
    parser.add_argument("--json-out", default="", help="optional output path for JSON summary")
    args = parser.parse_args()

    try:
        alerts = _load_alerts(args.repo, args.state)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = _summary(alerts)
    _print_human(summary)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
