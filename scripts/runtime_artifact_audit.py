from __future__ import annotations

import argparse
import json
from pathlib import Path

from eidosian_runtime.artifact_policy import (
    audit_runtime_artifacts,
    write_runtime_artifact_audit,
    write_runtime_artifact_audit_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit tracked and live runtime artifacts against policy.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--policy", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    if args.output:
        report = write_runtime_artifact_audit(args.repo_root, args.output, policy_path=args.policy)
    else:
        report = audit_runtime_artifacts(args.repo_root, policy_path=args.policy)
    if args.markdown_output:
        write_runtime_artifact_audit_markdown(report, args.markdown_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
