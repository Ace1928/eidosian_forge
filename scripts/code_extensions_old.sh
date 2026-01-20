#!/usr/bin/env bash
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Extensions aren't just tools—they're force multipliers ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Enforce strict execution for predictable behavior
set -euo pipefail

# Initialize extension state tracking with strict typing
declare -i INSTALLED_COUNT=0
declare -i FAILED_COUNT=0
declare -i SKIPPED_COUNT=0
declare -A CATEGORY_STATS=()
declare -a FAILED_EXTENSIONS=()
declare -a SKIPPED_EXTENSIONS=()
declare -a INSTALLATION_LOG=()

# Timestamp for operation benchmarking
START_TIME=$(date +%s)

# ╭────────────────────────────────────────────────────────────────────────╮
# │ EXTENSION MANAGEMENT PROTOCOL - PRECISE INSTALLATION FRAMEWORK         │
# ╰────────────────────────────────────────────────────────────────────────╯

# @signature: log_operation(String, String, String) -> Void
# @description: Logs operations with consistent formatting
# @param level: {String} Log level (INFO|WARN|ERROR)
# @param message: {String} Log message content
# @param details: {String} Optional additional details
log_operation() {
  local level="$1"
  local message="$2"
  local details="${3:-}"

  local timestamp
  timestamp=$(date "+%H:%M:%S")

  # Color coding for different log levels
  case "$level" in
    "INFO")  echo -e "\033[0;32m[INFO]\033[0m  $timestamp - $message $details" ;;
    "WARN")  echo -e "\033[0;33m[WARN]\033[0m  $timestamp - $message $details" ;;
    "ERROR") echo -e "\033[0;31m[ERROR]\033[0m $timestamp - $message $details" ;;
    *)       echo -e "$timestamp - $message $details" ;;
  esac

  # Store in log for final report
  INSTALLATION_LOG+=("[$level] $message $details")
}

# @signature: check_vscode_availability() -> Boolean
# @description: Verifies if VS Code CLI is available
check_vscode_availability() {
  if ! command -v code &> /dev/null; then
    log_operation "ERROR" "VS Code not found in PATH. Installation aborted."
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃ VS Code CLI not available. Please install VS Code      ┃"
    echo "┃ or ensure 'code' command is in your PATH.              ┃"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    exit 1
  fi
  return 0
}

# @signature: install_extension(String, String) -> Integer
# @description: Installs VSCode extension with elegant error handling and recovery
# @param extension_id: {String} Extension identifier to install
# @param category: {String} Functional category for extension
# @returns: {Integer} Status code: 0=success, 1=error, 2=skipped
install_extension() {
  local extension_id="$1"
  local category="${2:-uncategorized}"
  local extension_name
  extension_name=$(echo "$extension_id" | awk -F. '{print $NF}')

  # Initialize category counter if not exists
  CATEGORY_STATS["$category"]=${CATEGORY_STATS["$category"]:-0}

  # Check if extension is already installed to prevent redundant operations
  if code --list-extensions 2>/dev/null | grep -q "^$extension_id$"; then
    echo "⟳ ${extension_name} already installed."
    ((SKIPPED_COUNT++))
    SKIPPED_EXTENSIONS+=("$extension_name")
    return 2
  fi

  echo -n "⚡ Installing ${extension_name}... "

  if code --install-extension "$extension_id" &>/dev/null; then
    echo "✓"
    ((INSTALLED_COUNT++))
    ((CATEGORY_STATS["$category"]++))
    return 0
  else
    echo "✗"
    log_operation "ERROR" "Failed to install $extension_name" "($extension_id)"
    FAILED_EXTENSIONS+=("$extension_name")
    ((FAILED_COUNT++))
    return 1
  fi
}

# @signature: generate_extension_report() -> Void
# @description: Produces a comprehensive installation report with precise metrics
generate_extension_report() {
  local total_extensions
  total_extensions=$(code --list-extensions 2>/dev/null | wc -l)
  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - START_TIME))

  echo ""
  echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
  echo "┃ EXTENSION INSTALLATION REPORT                          ┃"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
  echo ""
  echo "✓ Successfully installed: $INSTALLED_COUNT extensions"
  echo "⟳ Already installed:     $SKIPPED_COUNT extensions"
  echo "✗ Installation failures: $FAILED_COUNT extensions"
  echo "⏱️  Total execution time:  ${duration}s"

  # Display category-based installation metrics
  echo ""
  echo "┌────────────────────────────────────────────────────────┐"
  echo "│ CATEGORY DISTRIBUTION                                  │"
  echo "└────────────────────────────────────────────────────────┘"
  for category in "${!CATEGORY_STATS[@]}"; do
    printf "%-15s : %3d extensions\n" "$category" "${CATEGORY_STATS[$category]}"
  done

  # Report failures with specific error messaging
  if [ "$FAILED_COUNT" -gt 0 ]; then
    echo ""
    echo "┌────────────────────────────────────────────────────────┐"
    echo "│ INSTALLATION FAILURES                                  │"
    echo "└────────────────────────────────────────────────────────┘"
    for extension in "${FAILED_EXTENSIONS[@]}"; do
      echo "  ✗ $extension"
    done
    echo ""
    echo "Run with '--verbose' to see detailed error logs."
  fi

  # System readiness assessment
  echo ""
  echo "┌────────────────────────────────────────────────────────┐"
  echo "│ SYSTEM STATUS                                          │"
  echo "└────────────────────────────────────────────────────────┘"
  echo "✓ Active extensions: $total_extensions total"

  # Add witty closing thought
  echo ""
  echo "┌────────────────────────────────────────────────────────┐"
  echo "│ ʕ•́ᴥ•̀ʔ \"Quality extensions are like good spices —     │"
  echo "│         essential in the right proportions.\"           │"
  echo "└────────────────────────────────────────────────────────┘"

  # Final system status with emoji for visual processing speed
  if [ "$FAILED_COUNT" -eq 0 ]; then
    echo "⚡ Environment initialization complete. Optimal tooling achieved."
  else
    echo "⚠️  Environment partially initialized with $FAILED_COUNT failures."
    echo "   Run installation again or investigate specific failures."
  fi
}

# ╭────────────────────────────────────────────────────────╮
# │ MAIN EXECUTION PROTOCOL                                │
# ╰────────────────────────────────────────────────────────╯

main() {
  # Verify VS Code is available before proceeding
  check_vscode_availability

  log_operation "INFO" "Extension installation initiated" "$(date)"

  # ╭────────────────────────────────────────────────────────╮
  # │ CORE FRAMEWORK                                         │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ INITIALIZING CORE FRAMEWORK                            │"
  echo "╰────────────────────────────────────────────────────────╯"

  # Material Icon Theme - Visual indexing for rapid pattern recognition
  install_extension "pkief.material-icon-theme" "core"

  # Error Lens - Immediate feedback and failure visualization
  install_extension "usernamehw.errorlens" "core"

  # GitLens - Source control and history tracking
  install_extension "eamodio.gitlens" "core"

  # Path Intellisense - Filesystem navigation assistant
  install_extension "christian-kohler.path-intellisense" "core"

  # ╭────────────────────────────────────────────────────────╮
  # │ SYNTAX STRUCTURE                                       │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ ESTABLISHING SYNTAX STRUCTURE                          │"
  echo "╰────────────────────────────────────────────────────────╯"

  # Prettier - Code formatting standardization
  install_extension "esbenp.prettier-vscode" "syntax"

  # ESLint - JavaScript/TypeScript linting
  install_extension "dbaeumer.vscode-eslint" "syntax"

  # Better Comments - Enhanced comment organization
  install_extension "aaron-bond.better-comments" "syntax"

  # Indent Rainbow - Nested structure visualization
  install_extension "oderwat.indent-rainbow" "syntax"

  # TODO Tree - Task and reminder management
  install_extension "Gruntfuggly.todo-tree" "syntax"

  # ╭────────────────────────────────────────────────────────╮
  # │ LANGUAGE-SPECIFIC TOOLS                                │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ DEPLOYING LANGUAGE-SPECIFIC TOOLS                      │"
  echo "╰────────────────────────────────────────────────────────╯"

  # === Rust Toolkit ===
  install_extension "rust-lang.rust-analyzer" "language"
  install_extension "tamasfe.even-better-toml" "language"

  # === TypeScript/JavaScript Framework ===
  install_extension "ms-vscode.vscode-typescript-next" "language"
  install_extension "dsznajder.es7-react-js-snippets" "language"
  install_extension "xabikos.JavaScriptSnippets" "language"
  install_extension "Angular.ng-template" "language"

  # === C/C++ Development ===
  install_extension "ms-vscode.cpptools" "language"
  install_extension "jeff-hykin.better-cpp-syntax" "language"
  install_extension "twxs.cmake" "language"

  # === Go Development ===
  install_extension "golang.go" "language"

  # === Python Development ===
  install_extension "ms-python.python" "language"
  install_extension "ms-python.vscode-pylance" "language"
  install_extension "njpwerner.autodocstring" "language"

  # === JVM Development ===
  install_extension "redhat.java" "language"
  install_extension "vscjava.vscode-java-debug" "language"
  install_extension "redhat.vscode-xml" "language"
  install_extension "fwcd.kotlin" "language"

  # === Web Development ===
  install_extension "ecmel.vscode-html-css" "language"
  install_extension "jock.svg" "language"
  install_extension "bradlc.vscode-tailwindcss" "language"
  install_extension "Vue.volar" "language"

  # === Documentation Tools ===
  install_extension "yzhang.markdown-all-in-one" "language"
  install_extension "redhat.vscode-yaml" "language"
  install_extension "TakumiI.markdowntable" "language"

  # === Command Line Tools ===
  install_extension "timonwong.shellcheck" "language"
  install_extension "ms-azuretools.vscode-docker" "language"

  # ╭────────────────────────────────────────────────────────╮
  # │ COLLABORATIVE TOOLS                                    │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ ESTABLISHING COLLABORATIVE FRAMEWORK                   │"
  echo "╰────────────────────────────────────────────────────────╯"

  # Remote Development Pack - Cross-environment development
  install_extension "ms-vscode-remote.vscode-remote-extensionpack" "collab"

  # Live Share - Real-time collaboration
  install_extension "ms-vsliveshare.vsliveshare" "collab"

  # GitLab Workflow - GitLab integration
  install_extension "gitlab.gitlab-workflow" "collab"

  # ╭────────────────────────────────────────────────────────╮
  # │ SYSTEM OPTIMIZATION                                    │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ DEPLOYING SYSTEM OPTIMIZATION TOOLS                    │"
  echo "╰────────────────────────────────────────────────────────╯"

  # Settings Sync - Configuration synchronization
  install_extension "Shan.code-settings-sync" "system"

  # Peacock - Workspace color differentiation
  install_extension "johnpapa.vscode-peacock" "system"

  # Import Cost - Dependency size visualization
  install_extension "wix.vscode-import-cost" "system"

  # Performance tools
  install_extension "ms-vscode.js-debug-nightly" "system"
  install_extension "streetsidesoftware.code-spell-checker" "system"

  # ╭────────────────────────────────────────────────────────╮
  # │ AESTHETIC ENHANCEMENTS                                 │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ CONFIGURING VISUAL INTERFACE                           │"
  echo "╰────────────────────────────────────────────────────────╯"

  # Theme options
  install_extension "dracula-theme.theme-dracula" "aesthetic"
  install_extension "GitHub.github-vscode-theme" "aesthetic"

  # ╭────────────────────────────────────────────────────────╮
  # │ AI ASSISTANCE                                          │
  # ╰────────────────────────────────────────────────────────╯

  echo "╭────────────────────────────────────────────────────────╮"
  echo "│ INTEGRATING AI ASSISTANCE                              │"
  echo "╰────────────────────────────────────────────────────────╯"

  # GitHub Copilot - AI code completion
  install_extension "GitHub.copilot" "ai"

  # Copilot Chat - AI conversation interface
  install_extension "GitHub.copilot-chat" "ai"

  # Generate final report
  generate_extension_report
}

# Execute main function with error handling
{
  main "$@"
} || {
  exit_code=$?
  log_operation "ERROR" "Script execution failed" "Exit code: $exit_code"
  echo "⚠️  Installation process interrupted. Some extensions may not be installed."
  exit $exit_code
}
