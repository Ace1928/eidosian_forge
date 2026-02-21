#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# Prefer project venv for consistent plugin resolution in Termux/Linux.
if [ -x "$ROOT_DIR/eidosian_venv/bin/python" ]; then
  PY="$ROOT_DIR/eidosian_venv/bin/python"
else
  PY=python3
fi

OUT_DIR=${OUT_DIR:-"$ROOT_DIR/reports/local_test_audit"}
DEFAULT_TIMEOUT=${DEFAULT_TIMEOUT:-300}
mkdir -p "$OUT_DIR"
SUMMARY_TSV="$OUT_DIR/summary.tsv"
: > "$SUMMARY_TSV"

forge_timeout() {
  case "$1" in
    agent_forge) echo "${AGENT_FORGE_TIMEOUT:-900}" ;;
    figlet_forge) echo "${FIGLET_FORGE_TIMEOUT:-900}" ;;
    *) echo "$DEFAULT_TIMEOUT" ;;
  esac
}

if [ "${FORGES:-}" ]; then
  FORGE_SET=$FORGES
else
  FORGE_SET=$(printf "%s\n" *_forge)
fi

for forge in $FORGE_SET; do
  [ -d "$forge/tests" ] || continue
  log="$OUT_DIR/$forge.log"
  timeout_s=$(forge_timeout "$forge")

  start=$(date +%s)
  if timeout "${timeout_s}s" "$PY" -m pytest -q "$forge/tests" --maxfail=1 >"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  end=$(date +%s)
  dur=$((end - start))

  printf "%s\t%s\t%s\t%s\n" "$forge" "$rc" "$dur" "$log" >> "$SUMMARY_TSV"

  if [ "$rc" -eq 0 ]; then
    status="PASS"
  elif [ "$rc" -eq 124 ]; then
    status="TIMEOUT"
  else
    status="FAIL"
  fi
  printf '[%s] %s %ss\n' "$status" "$forge" "$dur"
done

awk -F '\t' 'BEGIN{t=0;f=0} {t++; if($2!=0) f++} END{printf("TOTAL=%d FAIL=%d\n",t,f)}' "$SUMMARY_TSV"
awk -F '\t' '$2!=0 {printf(" - %s code=%s dur=%ss log=%s\n",$1,$2,$3,$4)}' "$SUMMARY_TSV"
