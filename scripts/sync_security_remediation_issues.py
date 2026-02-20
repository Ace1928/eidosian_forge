#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER_PREFIX = "<!-- eidos-security-remediation-batch:"
MARKER_SUFFIX = " -->"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_batch_key(body: str) -> str:
    text = str(body or "")
    start = text.find(MARKER_PREFIX)
    if start < 0:
        return ""
    offset = start + len(MARKER_PREFIX)
    end = text.find(MARKER_SUFFIX, offset)
    if end < 0:
        return ""
    return text[offset:end].strip()


def _render_issue_title(batch: dict[str, Any]) -> str:
    ecosystem = str(batch.get("ecosystem") or "unknown")
    manifest = str(batch.get("manifest_path") or "unknown")
    severity = str(batch.get("max_severity") or "unknown")
    return f"Security Remediation Batch [{severity}] {ecosystem} :: {manifest}"


def _render_issue_body(batch: dict[str, Any]) -> str:
    key = str(batch.get("batch_key") or "unknown")
    severity = str(batch.get("max_severity") or "unknown")
    ecosystem = str(batch.get("ecosystem") or "unknown")
    manifest = str(batch.get("manifest_path") or "unknown")
    total_alerts = int(batch.get("total_alerts") or 0)
    branch = str(batch.get("suggested_branch") or "")
    pr_title = str(batch.get("suggested_pr_title") or "")

    lines: list[str] = []
    lines.append(f"{MARKER_PREFIX}{key}{MARKER_SUFFIX}")
    lines.append("# Security Remediation Batch")
    lines.append("")
    lines.append(f"- batch_key: `{key}`")
    lines.append(f"- max_severity: `{severity}`")
    lines.append(f"- ecosystem: `{ecosystem}`")
    lines.append(f"- manifest_path: `{manifest}`")
    lines.append(f"- total_alerts: `{total_alerts}`")
    lines.append(f"- suggested_branch: `{branch}`")
    lines.append(f"- suggested_pr_title: `{pr_title}`")
    lines.append("")
    lines.append("## Checklist")
    lines.append("- [ ] Create remediation branch.")
    lines.append("- [ ] Upgrade vulnerable dependencies for this manifest.")
    lines.append("- [ ] Run module tests and CI checks.")
    lines.append("- [ ] Open PR with remediation notes and risk assessment.")
    lines.append("- [ ] Verify Dependabot alerts for this batch are resolved.")
    lines.append("")
    lines.append("## Top Packages")
    for pkg in batch.get("top_packages") or []:
        if not isinstance(pkg, dict):
            continue
        lines.append(
            "- `{name}` x `{count}`".format(
                name=pkg.get("name"),
                count=pkg.get("count"),
            )
        )
    lines.append("")
    lines.append("## Alerts (sample)")
    for row in batch.get("alerts") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- #{num} `{sev}` `{pkg}` {summary}".format(
                num=row.get("number"),
                sev=row.get("severity"),
                pkg=row.get("package"),
                summary=str(row.get("summary") or "").strip(),
            )
        )
    overflow = int(batch.get("overflow_alert_count") or 0)
    if overflow > 0:
        lines.append(f"- ... plus `{overflow}` additional alerts in this batch")
    lines.append("")
    lines.append(f"_Generated at `{_now_iso()}`_")
    return "\n".join(lines) + "\n"


def _api_json(
    method: str,
    url: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "eidosian-security-remediation-sync",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


def _list_open_issues(repo: str, token: str) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    page = 1
    out: list[dict[str, Any]] = []
    while True:
        url = (
            f"https://api.github.com/repos/{owner}/{name}/issues"
            f"?state=open&per_page=100&page={page}&sort=created&direction=asc"
        )
        payload = _api_json("GET", url, token=token)
        if not isinstance(payload, list):
            break
        rows = [x for x in payload if isinstance(x, dict)]
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 100:
            break
        page += 1
    return out


def _create_issue(repo: str, token: str, title: str, body: str) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/issues"
    return _api_json("POST", url, token=token, payload={"title": title, "body": body})


def _update_issue(repo: str, token: str, issue_number: int, title: str, body: str) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}"
    return _api_json("PATCH", url, token=token, payload={"title": title, "body": body})


def _close_issue(repo: str, token: str, issue_number: int) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}"
    return _api_json("PATCH", url, token=token, payload={"state": "closed"})


def _comment_issue(repo: str, token: str, issue_number: int, body: str) -> dict[str, Any]:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/comments"
    return _api_json("POST", url, token=token, payload={"body": body})


def _index_marked_issues(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        key = _extract_batch_key(str(issue.get("body") or ""))
        if not key:
            continue
        grouped.setdefault(key, []).append(issue)
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda row: int(row.get("number") or 0))
    return grouped


def sync_remediation_issues(
    *,
    repo: str,
    token: str,
    plan: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    desired_batches = [x for x in (plan.get("batches") or []) if isinstance(x, dict) and x.get("batch_key")]
    desired_by_key = {str(x.get("batch_key")): x for x in desired_batches}

    open_issues = _list_open_issues(repo, token)
    existing = _index_marked_issues(open_issues)

    created: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    duplicate_closed: list[dict[str, Any]] = []

    for key, batch in desired_by_key.items():
        title = _render_issue_title(batch)
        body = _render_issue_body(batch)
        issue_rows = list(existing.get(key) or [])
        primary = issue_rows[0] if issue_rows else None
        duplicates = issue_rows[1:] if len(issue_rows) > 1 else []

        if primary is None:
            row = {"key": key, "title": title, "action": "create"}
            if not dry_run:
                created_issue = _create_issue(repo, token, title, body)
                row["issue_number"] = int(created_issue.get("number") or 0)
            created.append(row)
        else:
            number = int(primary.get("number") or 0)
            current_title = str(primary.get("title") or "")
            current_body = str(primary.get("body") or "")
            if current_title != title or current_body != body:
                row = {"key": key, "issue_number": number, "action": "update"}
                if not dry_run:
                    _update_issue(repo, token, number, title, body)
                updated.append(row)
        for dup in duplicates:
            number = int(dup.get("number") or 0)
            row = {"key": key, "issue_number": number, "action": "close_duplicate"}
            if not dry_run and number > 0:
                _comment_issue(
                    repo,
                    token,
                    number,
                    "Closing duplicate remediation batch issue in favor of the canonical lowest-numbered issue.",
                )
                _close_issue(repo, token, number)
            duplicate_closed.append(row)

    stale_keys = sorted(set(existing.keys()) - set(desired_by_key.keys()))
    for key in stale_keys:
        issue_rows = existing.get(key) or []
        for row in issue_rows:
            number = int(row.get("number") or 0)
            close_row = {"key": key, "issue_number": number, "action": "close_stale"}
            if not dry_run and number > 0:
                _comment_issue(
                    repo,
                    token,
                    number,
                    "Closing remediation batch issue because current audit no longer includes this batch.",
                )
                _close_issue(repo, token, number)
            closed.append(close_row)

    return {
        "generated_at": _now_iso(),
        "repo": repo,
        "dry_run": bool(dry_run),
        "desired_batch_count": int(len(desired_by_key)),
        "existing_marked_batch_count": int(len(existing)),
        "created_count": int(len(created)),
        "updated_count": int(len(updated)),
        "closed_stale_count": int(len(closed)),
        "closed_duplicate_count": int(len(duplicate_closed)),
        "created": created,
        "updated": updated,
        "closed": closed,
        "closed_duplicates": duplicate_closed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync remediation batch issues from a remediation plan JSON.")
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--plan-json", required=True, help="Path to remediation plan JSON.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Env var containing GitHub token.")
    parser.add_argument("--json-out", default="", help="Optional sync output JSON path.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan_json).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    token = str(os.environ.get(args.token_env) or os.environ.get("GH_TOKEN") or "")
    if not token and not args.dry_run:
        raise SystemExit(f"missing token in env var '{args.token_env}'")

    report = sync_remediation_issues(
        repo=str(args.repo),
        token=token,
        plan=plan,
        dry_run=bool(args.dry_run),
    )
    print(
        "remediation_issue_sync: "
        f"desired={report.get('desired_batch_count')} "
        f"created={report.get('created_count')} "
        f"updated={report.get('updated_count')} "
        f"closed={report.get('closed_stale_count')} "
        f"duplicates={report.get('closed_duplicate_count')}"
    )
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
