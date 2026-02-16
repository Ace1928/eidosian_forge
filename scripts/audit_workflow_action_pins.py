#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

USES_RE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MUTABLE_REFS = {"main", "master", "head", "trunk", "latest"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sha_ref(ref: str) -> bool:
    return bool(SHA_RE.fullmatch(str(ref or "").strip().lower()))


def _is_mutable_ref(ref: str) -> bool:
    lowered = str(ref or "").strip().lower()
    if _is_sha_ref(lowered):
        return False
    if lowered in MUTABLE_REFS:
        return True
    if lowered.startswith("refs/heads/") or lowered.startswith("heads/"):
        return True
    return False


def _parse_action_spec(spec: str) -> dict[str, str] | None:
    raw = str(spec or "").strip().strip("'\"")
    if not raw or "@" not in raw:
        return None
    if raw.startswith("./") or raw.startswith("docker://"):
        return None
    action_path, ref = raw.rsplit("@", 1)
    parts = action_path.split("/")
    if len(parts) < 2:
        return None
    owner = parts[0].strip()
    repo = parts[1].strip()
    if not owner or not repo or not ref.strip():
        return None
    return {
        "spec": raw,
        "action_path": action_path,
        "owner": owner,
        "repo": repo,
        "ref": ref.strip(),
        "key": f"{action_path}@{ref.strip()}",
    }


def scan_workflow_actions(workflows_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not workflows_dir.exists():
        return rows
    files = sorted(list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")))
    for workflow in files:
        try:
            lines = workflow.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            match = USES_RE.search(line)
            if not match:
                continue
            parsed = _parse_action_spec(match.group(1))
            if parsed is None:
                continue
            rows.append(
                {
                    "workflow": str(workflow),
                    "line": int(line_no),
                    **parsed,
                }
            )
    return rows


def _fetch_json(url: str, *, token: str = "", timeout: float = 10.0) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "eidosian-workflow-action-audit",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_action_ref(
    owner: str,
    repo: str,
    ref: str,
    *,
    token: str = "",
    timeout: float = 10.0,
) -> tuple[str, str]:
    encoded_ref = urllib.parse.quote(str(ref), safe="")
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{encoded_ref}"
    try:
        payload = _fetch_json(url, token=token, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        return "", f"http_{exc.code}:{body[:200]}"
    except Exception as exc:
        return "", f"{type(exc).__name__}:{exc}"
    sha = str(payload.get("sha") or "").strip().lower()
    if not _is_sha_ref(sha):
        return "", "missing_sha"
    return sha, ""


def _load_lock(lock_file: Path) -> dict[str, str]:
    if not lock_file.exists():
        return {}
    try:
        payload = json.loads(lock_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("pins"), dict):
        return {str(k): str(v).strip().lower() for k, v in payload["pins"].items() if str(v).strip()}
    if isinstance(payload, dict):
        out: dict[str, str] = {}
        for key, value in payload.items():
            if key.startswith("_"):
                continue
            sval = str(value).strip().lower()
            if sval:
                out[str(key)] = sval
        return out
    return {}


def _write_lock(lock_file: Path, pins: dict[str, str], *, workflows_dir: Path) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generated_at": _now_iso(),
            "workflows_dir": str(workflows_dir),
            "entries": len(pins),
        },
        "pins": dict(sorted(pins.items())),
    }
    lock_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_audit_report(
    usages: list[dict[str, Any]],
    *,
    lock_pins: dict[str, str],
    resolver: Callable[[str, str, str], tuple[str, str]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in usages:
        key = str(row.get("key") or "")
        if not key:
            continue
        if key not in grouped:
            grouped[key] = {
                "base": row,
                "locations": [],
            }
        grouped[key]["locations"].append(
            {
                "workflow": str(row.get("workflow") or ""),
                "line": int(row.get("line") or 0),
            }
        )

    cache: dict[tuple[str, str, str], tuple[str, str]] = {}
    entries: list[dict[str, Any]] = []
    resolved_pins: dict[str, str] = {}

    for key in sorted(grouped.keys()):
        base = grouped[key]["base"]
        owner = str(base.get("owner") or "")
        repo = str(base.get("repo") or "")
        ref = str(base.get("ref") or "")
        cache_key = (owner, repo, ref)
        if cache_key not in cache:
            cache[cache_key] = resolver(owner, repo, ref)
        sha, err = cache[cache_key]
        lock_sha = str(lock_pins.get(key) or "").strip().lower()
        drift = bool(lock_sha and sha and lock_sha != sha)
        unlocked = key not in lock_pins
        mutable_ref = _is_mutable_ref(ref)
        sha_pinned = _is_sha_ref(ref)
        if sha:
            resolved_pins[key] = sha

        entries.append(
            {
                "key": key,
                "action_path": str(base.get("action_path") or ""),
                "owner": owner,
                "repo": repo,
                "ref": ref,
                "sha_pinned_ref": sha_pinned,
                "mutable_ref": mutable_ref,
                "resolved_sha": sha,
                "lock_sha": lock_sha,
                "drift": drift,
                "unlocked": unlocked,
                "resolve_error": err,
                "locations": grouped[key]["locations"],
                "occurrences": len(grouped[key]["locations"]),
            }
        )

    stale_lock = sorted(set(lock_pins.keys()) - set(grouped.keys()))
    summary = {
        "workflows_scanned": len(sorted({str(x.get("workflow") or "") for x in usages})),
        "uses_entries_scanned": len(usages),
        "unique_action_refs": len(entries),
        "sha_pinned_ref_count": sum(1 for x in entries if x["sha_pinned_ref"]),
        "mutable_ref_count": sum(1 for x in entries if x["mutable_ref"]),
        "resolved_count": sum(1 for x in entries if x["resolved_sha"]),
        "unresolved_count": sum(1 for x in entries if not x["resolved_sha"]),
        "drift_count": sum(1 for x in entries if x["drift"]),
        "unlocked_count": sum(1 for x in entries if x["unlocked"]),
        "stale_lock_count": len(stale_lock),
    }

    return {
        "generated_at": _now_iso(),
        "summary": summary,
        "entries": entries,
        "stale_lock_entries": stale_lock,
        "resolved_pins": dict(sorted(resolved_pins.items())),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit GitHub Actions workflow action references and enforce lock drift policy."
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument("--lock-file", default="audit_data/workflow_action_lock.json")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--enforce-lock", action="store_true")
    parser.add_argument("--fail-on-mutable", action="store_true")
    parser.add_argument("--strict-resolve", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    workflows_dir = Path(args.workflows_dir).expanduser().resolve()
    lock_file = Path(args.lock_file).expanduser().resolve()
    token = str(os.environ.get(args.token_env) or os.environ.get("GH_TOKEN") or "")

    usages = scan_workflow_actions(workflows_dir)
    lock_pins = _load_lock(lock_file)

    def _resolver(owner: str, repo: str, ref: str) -> tuple[str, str]:
        return resolve_action_ref(owner, repo, ref, token=token, timeout=max(1.0, float(args.timeout)))

    report = build_audit_report(usages, lock_pins=lock_pins, resolver=_resolver)
    summary = report.get("summary") or {}

    print(
        "workflow_action_audit: "
        f"workflows={summary.get('workflows_scanned', 0)} "
        f"unique_refs={summary.get('unique_action_refs', 0)} "
        f"resolved={summary.get('resolved_count', 0)} "
        f"drift={summary.get('drift_count', 0)} "
        f"unlocked={summary.get('unlocked_count', 0)} "
        f"mutable={summary.get('mutable_ref_count', 0)}"
    )

    if args.report_json:
        _write_report(Path(args.report_json).expanduser().resolve(), report)

    if args.update_lock:
        current_pins = dict(lock_pins)
        for key, sha in (report.get("resolved_pins") or {}).items():
            if sha:
                current_pins[str(key)] = str(sha)
        active_keys = {str(entry.get("key") or "") for entry in (report.get("entries") or [])}
        current_pins = {k: v for k, v in current_pins.items() if k in active_keys and v}
        _write_lock(lock_file, current_pins, workflows_dir=workflows_dir)
        print(f"updated lock file: {lock_file}")

    failures: list[str] = []
    if args.fail_on_mutable and int(summary.get("mutable_ref_count", 0)) > 0:
        failures.append("mutable_refs_detected")
    if args.enforce_lock:
        if int(summary.get("drift_count", 0)) > 0:
            failures.append("lock_drift_detected")
        if int(summary.get("unlocked_count", 0)) > 0:
            failures.append("unlocked_action_refs_detected")
        if args.strict_resolve and int(summary.get("unresolved_count", 0)) > 0:
            failures.append("unresolved_action_refs_detected")

    if failures:
        print("policy_failures: " + ", ".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
