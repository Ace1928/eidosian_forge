#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
    "unknown": 0,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _severity(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in SEVERITY_ORDER else "unknown"


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    out = out.strip("-")
    return out or "unknown"


def _batch_key(ecosystem: str, manifest_path: str) -> str:
    return f"{_slug(ecosystem)}::{_slug(manifest_path)}"


def _alert_number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_remediation_plan(
    *,
    summary: dict[str, Any],
    repo: str,
    include_severities: set[str],
    max_batches: int = 30,
    min_alerts_per_batch: int = 1,
    max_alerts_per_batch: int = 40,
) -> dict[str, Any]:
    rows = summary.get("open_high_critical")
    alerts = [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in alerts:
        sev = _severity(row.get("severity"))
        if sev not in include_severities:
            continue
        ecosystem = str(row.get("ecosystem") or "unknown")
        manifest_path = str(row.get("manifest_path") or "unknown")
        grouped[(ecosystem, manifest_path)].append(row)

    batches: list[dict[str, Any]] = []
    for (ecosystem, manifest_path), items in grouped.items():
        if len(items) < max(1, int(min_alerts_per_batch)):
            continue
        severity_counts = Counter(_severity(x.get("severity")) for x in items)
        max_sev = max(
            (sev for sev in severity_counts.keys()),
            key=lambda sev: SEVERITY_ORDER.get(sev, 0),
            default="unknown",
        )
        package_counts = Counter(str(x.get("package") or "unknown") for x in items)
        sorted_items = sorted(
            items,
            key=lambda row: (
                -SEVERITY_ORDER.get(_severity(row.get("severity")), 0),
                str(row.get("package") or ""),
                _alert_number(row.get("number")),
            ),
        )
        capped = sorted_items[: max(1, int(max_alerts_per_batch))]
        total = len(items)
        priority_score = (
            int(severity_counts.get("critical", 0)) * 1000
            + int(severity_counts.get("high", 0)) * 100
            + int(total)
        )
        top_packages = [
            {"name": name, "count": int(count)}
            for name, count in package_counts.most_common(10)
        ]
        alert_rows = [
            {
                "number": _alert_number(x.get("number")),
                "severity": _severity(x.get("severity")),
                "package": str(x.get("package") or "unknown"),
                "summary": str(x.get("summary") or "").strip(),
            }
            for x in capped
        ]
        batch_key = _batch_key(ecosystem, manifest_path)
        branch_slug = _slug(f"{ecosystem}-{manifest_path}-{max_sev}")
        batches.append(
            {
                "batch_key": batch_key,
                "repo": repo,
                "ecosystem": ecosystem,
                "manifest_path": manifest_path,
                "max_severity": max_sev,
                "severity_counts": {
                    "critical": int(severity_counts.get("critical", 0)),
                    "high": int(severity_counts.get("high", 0)),
                    "moderate": int(severity_counts.get("moderate", 0)),
                    "low": int(severity_counts.get("low", 0)),
                    "unknown": int(severity_counts.get("unknown", 0)),
                },
                "total_alerts": int(total),
                "package_count": int(len(package_counts)),
                "top_packages": top_packages,
                "alerts": alert_rows,
                "overflow_alert_count": max(0, int(total - len(alert_rows))),
                "priority_score": int(priority_score),
                "suggested_branch": f"security/remediation/{branch_slug}",
                "suggested_pr_title": (
                    f"security: remediate {max_sev} deps in {manifest_path}"
                ),
            }
        )

    batches = sorted(
        batches,
        key=lambda x: (
            -int(x.get("priority_score") or 0),
            str(x.get("manifest_path") or ""),
            str(x.get("ecosystem") or ""),
        ),
    )[: max(1, int(max_batches))]

    return {
        "generated_at": _now_iso(),
        "repo": repo,
        "summary_error": str(summary.get("error") or ""),
        "batch_count": int(len(batches)),
        "total_batched_alerts": int(sum(int(x.get("total_alerts") or 0) for x in batches)),
        "include_severities": sorted(include_severities),
        "batches": batches,
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dependabot Remediation Batches")
    lines.append("")
    lines.append(f"- generated_at: `{plan.get('generated_at')}`")
    lines.append(f"- repo: `{plan.get('repo')}`")
    lines.append(f"- batch_count: `{plan.get('batch_count')}`")
    lines.append(f"- total_batched_alerts: `{plan.get('total_batched_alerts')}`")
    if plan.get("summary_error"):
        lines.append(f"- summary_error: `{plan.get('summary_error')}`")
    lines.append("")

    batches = plan.get("batches") if isinstance(plan.get("batches"), list) else []
    if not batches:
        lines.append("No remediation batches generated.")
        return "\n".join(lines) + "\n"

    lines.append("## Batches")
    lines.append("")
    lines.append("| Key | Severity | Alerts | Ecosystem | Manifest |")
    lines.append("|---|---:|---:|---|---|")
    for row in batches:
        lines.append(
            "| `{key}` | `{sev}` | `{alerts}` | `{eco}` | `{manifest}` |".format(
                key=row.get("batch_key"),
                sev=row.get("max_severity"),
                alerts=row.get("total_alerts"),
                eco=row.get("ecosystem"),
                manifest=row.get("manifest_path"),
            )
        )
    lines.append("")

    for row in batches:
        lines.append(f"### `{row.get('batch_key')}`")
        lines.append("")
        lines.append(f"- max_severity: `{row.get('max_severity')}`")
        lines.append(f"- total_alerts: `{row.get('total_alerts')}`")
        lines.append(f"- package_count: `{row.get('package_count')}`")
        lines.append(f"- suggested_branch: `{row.get('suggested_branch')}`")
        lines.append(f"- suggested_pr_title: `{row.get('suggested_pr_title')}`")
        lines.append("")
        lines.append("Top packages:")
        top = row.get("top_packages") if isinstance(row.get("top_packages"), list) else []
        for pkg in top:
            lines.append(
                "- `{name}` x `{count}`".format(
                    name=pkg.get("name"),
                    count=pkg.get("count"),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate remediation batches from Dependabot summary JSON.")
    parser.add_argument("--summary-json", required=True, help="Path to dependabot summary JSON.")
    parser.add_argument("--repo", default="Ace1928/eidosian_forge", help="owner/repo identifier.")
    parser.add_argument(
        "--severity",
        default="critical,high",
        help="Comma-separated severities to include (default: critical,high).",
    )
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--min-alerts-per-batch", type=int, default=1)
    parser.add_argument("--max-alerts-per-batch", type=int, default=40)
    parser.add_argument("--json-out", default="", help="Optional output path for plan JSON.")
    parser.add_argument("--md-out", default="", help="Optional output path for markdown report.")
    args = parser.parse_args()

    summary_path = Path(args.summary_json).expanduser().resolve()
    summary = _load_json(summary_path)
    include_severities = {
        _severity(item)
        for item in str(args.severity or "critical,high").split(",")
        if str(item).strip()
    }
    include_severities.discard("unknown")
    if not include_severities:
        include_severities = {"critical", "high"}

    plan = build_remediation_plan(
        summary=summary,
        repo=str(args.repo),
        include_severities=include_severities,
        max_batches=max(1, int(args.max_batches)),
        min_alerts_per_batch=max(1, int(args.min_alerts_per_batch)),
        max_alerts_per_batch=max(1, int(args.max_alerts_per_batch)),
    )

    print(
        "dependabot_remediation_plan: "
        f"batches={plan.get('batch_count')} "
        f"alerts={plan.get('total_batched_alerts')} "
        f"severities={','.join(plan.get('include_severities') or [])}"
    )

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    if args.md_out:
        out_md = Path(args.md_out).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(plan), encoding="utf-8")
        print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
