#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROFILE_DIR/.." && pwd)"

run_time() {
  local name="$1"
  shift
  /usr/bin/time -p "$@" >"$PROFILE_DIR/${name}.out" 2>"$PROFILE_DIR/${name}.time"
}

run_cprofile() {
  local name="$1"
  shift
  python3 -m cProfile -o "$PROFILE_DIR/${name}.prof" "$@" >"$PROFILE_DIR/${name}.out" 2>"$PROFILE_DIR/${name}.err" || true
}

cd "$ROOT_DIR"

run_time "add_script_help" ./add_script --help
run_time "add_to_path_help" ./add_to_path --help
run_time "clipboard_diagnostics_help" ./clipboard_diagnostics.sh --help
run_time "clipboard_overwrite_help" ./clipboard_overwrite.sh --help
run_time "completion_install_help" ./completion_install --help
run_time "interactive_delete_help" ./interactive_delete --help
run_time "overwrite_help" ./overwrite --help

run_cprofile "pdf_check_text_help" ./pdf_check_text --help
run_cprofile "pdf_check_encryption_help" ./pdf_check_encryption --help
run_cprofile "pdf_to_text_help" ./pdf_to_text --help
run_cprofile "pdf_tools_menu_help" ./pdf_tools_menu --help
run_cprofile "smart_publish_help" ./smart_publish --help
run_cprofile "forge_builder_help" ./forge_builder --help
run_cprofile "glyph_stream_help" ./glyph_stream --help
run_cprofile "treegen_help" ./treegen --help

# Lightweight real run for treegen
run_time "treegen_lib" ./treegen --no-color --no-unicode --no-progress --output /tmp/treegen_out.txt --json-summary ./lib
