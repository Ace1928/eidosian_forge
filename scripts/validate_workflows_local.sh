#!/usr/bin/env sh
set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY_BIN="${PY_BIN:-./eidosian_venv/bin/python}"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

ACTIONLINT_BIN="$(go env GOPATH 2>/dev/null || true)/bin/actionlint"
if [ ! -x "$ACTIONLINT_BIN" ]; then
  echo "[workflows] installing actionlint"
  go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.11
fi

if [ ! -x "$(go env GOPATH)/bin/gitleaks" ]; then
  echo "[workflows] installing gitleaks"
  go install github.com/zricethezav/gitleaks/v8@v8.24.2
fi

REPORT_PATH="reports/workflow_action_pin_audit_local.json"
cleanup() {
  rm -f "$REPORT_PATH"
}
trap cleanup EXIT INT TERM

echo "[workflows] actionlint"
"$(go env GOPATH)/bin/actionlint" -color -oneline .github/workflows/*.yml

echo "[workflows] action pin lock audit"
PYTHONPATH=lib "$PY_BIN" scripts/audit_workflow_action_pins.py \
  --workflows-dir .github/workflows \
  --lock-file audit_data/workflow_action_lock.json \
  --report-json "$REPORT_PATH" \
  --enforce-lock \
  --fail-on-mutable

echo "[workflows] workflow support unit tests"
PYTHONPATH=lib "$PY_BIN" -m pytest -q \
  scripts/tests/test_audit_workflow_action_pins.py \
  scripts/tests/test_dependabot_remediation_plan.py \
  scripts/tests/test_dependabot_autopatch_requirements.py \
  scripts/tests/test_generate_directory_atlas.py \
  scripts/tests/test_consciousness_benchmark_trend.py \
  scripts/tests/test_linux_audit_matrix.py

echo "[workflows] secret scan sanity (.github scope)"
"$(go env GOPATH)/bin/gitleaks" dir .github --config=.gitleaks.toml --no-banner --redact --max-target-megabytes 5

echo "[workflows] atlas generation determinism"
before_atlas_hash="$(sha256sum docs/DIRECTORY_ATLAS.md 2>/dev/null | awk '{print $1}')"
before_index_hash="$(sha256sum docs/DIRECTORY_INDEX_FULL.txt 2>/dev/null | awk '{print $1}')"
PYTHONPATH=lib "$PY_BIN" scripts/generate_directory_atlas.py \
  --repo-root . \
  --atlas-output docs/DIRECTORY_ATLAS.md \
  --full-output docs/DIRECTORY_INDEX_FULL.txt \
  --max-depth 2 \
  --scope tracked
after_atlas_hash="$(sha256sum docs/DIRECTORY_ATLAS.md | awk '{print $1}')"
after_index_hash="$(sha256sum docs/DIRECTORY_INDEX_FULL.txt | awk '{print $1}')"
if [ "$before_atlas_hash" != "$after_atlas_hash" ] || [ "$before_index_hash" != "$after_index_hash" ]; then
  echo "[workflows] atlas artifacts changed; commit updated docs/DIRECTORY_* outputs"
fi

echo "[workflows] local workflow validation complete"
