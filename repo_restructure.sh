#!/usr/bin/env bash
# ╭──────────────────────────────────────────────────────────────────────╮
# │ 🏗️  EIDOSIAN FORGE - REPOSITORY RESTRUCTURER v1.0.3                  │
# │    Transforming chaos into crystalline architecture                  │
# │    With atomic precision and recursive elegance                      │
# ╰──────────────────────────────────────────────────────────────────────╯

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ EXECUTION PROTOCOL - DETERMINISTIC FAILURE BOUNDARIES                ┃
# ┃ Establishes atomistic execution context with explicit failure modes  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Validate shell compatibility - fail fast with clear remediation path
# @requires: Bash 4.0+ for associative arrays and parameter expansion
# @returns: {Integer} 0 on success, exits with code 1 on version mismatch
# @pure: False (has side effects - may exit process)
# @throws: Exits with code 1 and diagnostic message on version incompatibility
function validate_bash_version() {
  local -r min_major=4
  local -r current_major="${BASH_VERSINFO[0]}"

  if ((current_major < min_major)); then
    echo "⚠️  Fatal: Requires Bash ${min_major}.0+ for associative arrays and advanced features" >&2
    echo "→ Current version: ${BASH_VERSION}" >&2
    echo "→ Remediation: Use a compatible shell or upgrade bash" >&2
    return 1
  fi
  return 0
}

validate_bash_version || exit 1

# Establish deterministic failure modes with explicit boundary conditions
# @effect: Immediately exits on errors, undefined variables, and pipeline failures
# @rationale: Prevents silent failures, state corruption, and zombie processes
# @reliability: Guarantees execution integrity or immediate termination
set -euo pipefail

# Translate exit codes to human-readable meanings with precision
# @param: {Integer} Exit code to interpret
# @returns: {String} Human-readable description with semantic meaning
# @usage: meaning=$(exit_code_meaning 127)
# @pure: True (no side effects, deterministic output)
# @throws: Error if required parameter is missing
exit_code_meaning() {
  local -r code=${1:?Missing required exit code parameter}

  # Exit code dictionary with complete semantic mapping
  local -r exit_codes=(
    [0]="Success"
    [1]="General error"
    [2]="Misuse of shell builtins"
    [126]="Command invoked cannot execute"
    [127]="Command not found"
    [128]="Invalid argument to exit"
    [129]="SIGHUP - Hangup"
    [130]="SIGINT - Terminal interrupt"
    [131]="SIGQUIT - Terminal quit"
    [132]="SIGILL - Illegal instruction"
    [133]="SIGTRAP - Trace/breakpoint trap"
    [134]="SIGABRT - Abort"
    [135]="SIGBUS - Bus error"
    [136]="SIGFPE - Floating point exception"
    [137]="SIGKILL - Kill"
    [139]="SIGSEGV - Segmentation fault"
    [141]="SIGPIPE - Broken pipe"
    [143]="SIGTERM - Termination"
  )

  # Deterministic response with graceful fallback
  echo "${exit_codes[$code]:-Unknown error code: $code}"
}

# Create execution context for sophisticated error handling
# @detects: Exact failure point with complete contextual information
# @provides: Self-documenting error reports with actionable insights
# @returns: Non-zero exit code from the actual failure point
# @preserves: Original error code for proper error propagation
trap 'error_code=$?;
      cmd=${BASH_COMMAND/#$HOME/\~};
      echo "┏━━ ⚠️  EXECUTION BOUNDARY VIOLATION ━━━━━━━━━━━━━━━━━━━━━━";
      echo "┃ Location: Line $LINENO in $(basename "${BASH_SOURCE[0]}")";
      echo "┃ Command: ${cmd}";
      echo "┃ Exit Code: $error_code ($(exit_code_meaning $error_code))";
      echo "┃ Working Directory: $(pwd)";
      echo "┃ User: $(whoami)@$(hostname -s)";
      echo "┃ Time: $(date "+%Y-%m-%d %H:%M:%S")";
      echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
      exit $error_code' ERR

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ VERSION IDENTITY - IMMUTABLE EXECUTION CONTEXT                      ┃
# ┃ Establishes the foundational ontological parameters of execution    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Core identity markers - immutable execution context parameters
# @purpose: Provides consistent versioning and runtime identification
# @immutable: True (values should never be modified during execution)
# @type: String constants for deterministic reference
readonly SCRIPT_VERSION="1.0.3"                                                                                         # Semantic version with backward compatibility guarantees
readonly SCRIPT_NAME="$(basename "${0%.sh}")"                                                                           # Self-aware identifier with extension normalization
readonly SCRIPT_START_TIME="$(date +%s)"                                                                                # Execution time anchor for duration calculations and telemetry
readonly SCRIPT_PATH="$(readlink -f "$0")"                                                                              # Canonical path with symlink resolution for absolute reference
readonly SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"                                                                         # Execution context directory for relative resource location
readonly SCRIPT_HASH="$([ -f "$SCRIPT_PATH" ] && md5sum "$SCRIPT_PATH" 2> /dev/null | cut -d' ' -f1 || echo "unknown")" # Integrity verification

# ╭──────────────────────────────────────────────────────────────────────╮
# │ DEPENDENCY VALIDATION - ENVIRONMENT INTEGRITY VERIFICATION            │
# │ Ensures all external requirements are satisfied before execution      │
    "mkdir:directory creation"
    "tput:terminal capability detection"
    "locale:character encoding discovery"
    "grep:pattern matching"
    "date:temporal reference"
    "basename:path component extraction"
    "md5sum:integrity verification"
  )
  local -a missing_deps=()
  local -a version_issues=()

  # Deterministic validation protocol with isolation per dependency
  # Guarantees complete discovery of all issues before reporting
  for cmd_spec in "${REQUIRED_COMMANDS[@]}"; do
    local cmd="${cmd_spec%%:*}"    # Extract command name before colon
    local purpose="${cmd_spec#*:}" # Extract purpose description after colon

    if ! command -v "$cmd" > /dev/null 2>&1; then
      missing_deps+=("$cmd ($purpose)")
    elif [[ "$cmd" == "grep" ]] && ! grep --version | grep -q "GNU grep" > /dev/null 2>&1; then
      # Validate specific tool implementations when required
      version_issues+=("$cmd: requires GNU grep")
    fi
  done

  # Binary outcome with explicit remediation path and categorical issue separation
  local has_errors=0

  if [[ ${#missing_deps[@]} -gt 0 || ${#version_issues[@]} -gt 0 ]]; then
    {
      echo "╭─────────────────────────────────────────────────────╮"
      echo "│ ✗ ENVIRONMENT VALIDATION FAILED                     │"
      echo "├─────────────────────────────────────────────────────┤"

      # Missing dependencies report with purpose documentation
      if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo "│ Missing required tools:                             │"
        for dep in "${missing_deps[@]}"; do
          echo "│  • $dep"
        done
      fi

      # Version incompatibility report with specific requirements
      if [[ ${#version_issues[@]} -gt 0 ]]; then
        echo "│ Version compatibility issues:                       │"
        for issue in "${version_issues[@]}"; do
          echo "│  • $issue"
        done
      fi

      # Actionable remediation instructions with platform awareness
      if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        case "$ID" in
          debian | ubuntu)
            echo "│                                                     │"
            echo "│ Remediation (Debian/Ubuntu):                        │"
            echo "│ → sudo apt update && sudo apt install coreutils grep│"
            ;;
          fedora | rhel | centos)
            echo "│                                                     │"
            echo "│ Remediation (Fedora/RHEL/CentOS):                  │"
            echo "│ → sudo dnf install coreutils grep                   │"
            ;;
        esac
      else
        echo "│                                                     │"
        echo "│ Remediation:                                        │"
        echo "│ → Install missing tools with your package manager   │"
      fi

      echo "╰─────────────────────────────────────────────────────╯"
    } >&2
    has_errors=1
  fi

  return $has_errors # Explicit status code for precise error handling
}

# Execute validation immediately with fail-fast principle
# Early validation prevents cascading errors from missing dependencies
verify_runtime_dependencies || exit 1

# ╭──────────────────────────────────────────────────────────────────────╮
# │ TERMINAL DETECTION - ADAPTIVE RENDERING CONTEXT                      │
# │ Discovers environmental capabilities for optimal user experience     │
# ╰──────────────────────────────────────────────────────────────────────╯

# Terminal capability detection with adaptive degradation pathways
# Returns normalized boolean expressions for consistent evaluation across contexts
readonly HAS_TERMINAL="[[ -t 1 && -n \"$TERM\" && \"$TERM\" != \"dumb\" ]]"                                    # Interactive terminal detection
readonly TERMINAL_SUPPORTS_COLOR="$HAS_TERMINAL && tput colors >/dev/null 2>&1 && [[ \$(tput colors) -ge 8 ]]" # Color capability with minimum depth
readonly TERMINAL_SUPPORTS_UNICODE="$HAS_TERMINAL && locale charmap 2>/dev/null | grep -q 'UTF-\|utf-'"        # Unicode rendering support
readonly TERMINAL_WIDTH="$(eval "$HAS_TERMINAL" && tput cols 2> /dev/null || echo 80)"                         # Responsive layout support with fallback
readonly TERMINAL_HEIGHT="$(eval "$HAS_TERMINAL" && tput lines 2> /dev/null || echo 24)"                       # Vertical space awareness with fallback

# ╭──────────────────────────────────────────────────────────────────────╮
# │ TYPOGRAPHY PROTOCOL - ADAPTIVE VISUAL GRAMMAR                        │
# │ Establishes consistent visual hierarchy with graceful degradation    │
# ╰──────────────────────────────────────────────────────────────────────╯

# Terminal styles with graceful degradation cascade
# Form adapts to capability while preserving semantic meaning across environments
readonly STYLE_BOLD="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput bold 2> /dev/null || echo '')"
readonly STYLE_NORMAL="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput sgr0 2> /dev/null || echo '')"
readonly STYLE_UNDERLINE="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput smul 2> /dev/null || echo '')"
readonly STYLE_RESET="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput sgr0 2> /dev/null || echo '')"
readonly STYLE_DIM="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput dim 2> /dev/null || echo '')"
# Extended typography for semantic richness
readonly STYLE_ITALIC="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput sitm 2> /dev/null || echo '')"
readonly STYLE_BLINK="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput blink 2> /dev/null || echo '')" # Use sparingly - for critical alerts only
readonly STYLE_REVERSE="$(eval "$TERMINAL_SUPPORTS_COLOR" && tput rev 2> /dev/null || echo '')" # For selection highlighting

# ╭──────────────────────────────────────────────────────────────────────╮
# │ SEMANTIC INDICATORS - UNIVERSAL COMMUNICATION SYMBOLS                 │
# │ Provides consistent visual language across terminal capabilities      │
# ╰──────────────────────────────────────────────────────────────────────╯

# Status indicators with universal compatibility - semantic meaning persists across environments
# Color-coding follows intuitive psychological associations and accessibility standards
readonly ICON_INFO="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;36m➜\033[0m" || echo "INFO:")"            # Cyan: informational
readonly ICON_SUCCESS="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;32m✓\033[0m" || echo "SUCCESS:")"      # Green: successful completion
readonly ICON_WARNING="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;33m⚠\033[0m" || echo "WARNING:")"      # Yellow: warning condition
readonly ICON_ERROR="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;31m✗\033[0m" || echo "ERROR:")"          # Red: error condition
readonly ICON_DEBUG="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;35m◉\033[0m" || echo "DEBUG:")"          # Magenta: debug information
readonly ICON_QUESTION="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;34m?\033[0m" || echo "QUERY:")"       # Blue: interactive prompt
readonly ICON_WAITING="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;33m⋯\033[0m" || echo "WAIT:")"         # Yellow: processing indicator
readonly ICON_CRITICAL="$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo -e "\033[1;37;41m!\033[0m" || echo "CRITICAL:")" # White on red: critical alert

# ╭──────────────────────────────────────────────────────────────────────╮
# │ RUNTIME CAPABILITIES - EXECUTION CONTEXT AWARENESS                   │
# │ Identifies available features for optimal execution path selection   │
# ╰──────────────────────────────────────────────────────────────────────╯

# Feature flags - typed capability registry with introspection support
# Enables conditional logic based on available system features
declare -ra FEATURES=(
  "COLOR:$(eval "$TERMINAL_SUPPORTS_COLOR" && echo true || echo false)"           # Visual hierarchy through color
  "UNICODE:$(eval "$TERMINAL_SUPPORTS_UNICODE" && echo true || echo false)"       # Enhanced symbolic density
  "SCRIPT_VERSION:$SCRIPT_VERSION"                                                # Version tracking for compatibility
  "BASH_VERSION:${BASH_VERSION}"                                                  # Shell environment version
  "RESPONSIVE:$((TERMINAL_WIDTH > 80 ? 1 : 0))"                                   # Layout adaptation capability
  "ADMIN:$(eval "$HAS_ADMIN" && echo true || echo false)"                         # Elevated privileges detection
  "NETWORK:$(ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1 && echo true || echo false)" # Network connectivity detection
  "INTERACTIVE:$(eval "$HAS_TERMINAL" && echo true || echo false)"                # User interaction capability
)

# Query runtime features with type safety
# @usage: has_feature FEATURE_NAME
# @param: {String} Feature name to check
# @returns: 0 if feature is available, 1 if not available
# @example: if has_feature "COLOR"; then echo "Color support available"; fi
has_feature() {
  local feature_name="${1:?Missing required feature name}"
  local feature_value=""

  for feature in "${FEATURES[@]}"; do
    if [[ "${feature%%:*}" == "$feature_name" ]]; then
      feature_value="${feature#*:}"
      break
    fi
  done

  [[ "$feature_value" == "true" || "$feature_value" == "1" ]]
  return $?
}

# ┌────────────────────────────────────────────────────────────┐
# │ DOMAIN DISCOVERY - RUNTIME ENVIRONMENT INTROSPECTION        │
# │ Identifies execution context for adaptive behavior          │
# └────────────────────────────────────────────────────────────┘

# Detect execution environment characteristics for adaptive behavior
# These expressions evaluate lazily when used, preventing unnecessary probing
readonly IS_CI="[[ -n \"\${CI:-}\" || -n \"\${GITHUB_ACTIONS:-}\" || -n \"\${JENKINS_URL:-}\" || -n \"\${TRAVIS:-}\" ]]"
readonly IS_RESTRICTED_ENV="[[ -n \"\${RESTRICTED_ENV:-}\" || ! -w \"/tmp\" ]]" # Detect write-restricted environments
readonly HAS_ADMIN="[[ \$(id -u) -eq 0 ]]"                                      # Root/admin detection for privileged operations
readonly IS_CONTAINER="[[ -f \"/.dockerenv\" || -f \"/run/.containerenv\" || grep -q 'container=' /proc/1/environ 2>/dev/null ]]"
readonly OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')" # Operating system type for platform-specific behavior
readonly MACHINE_TYPE="$(uname -m)"                         # Architecture detection for binary compatibility

# ╭──────────────────────────────────────────────────────────────────────╮
# │ COMMUNICATION PROTOCOL - TYPED MESSAGE EXCHANGE SYSTEM                │
# ╰──────────────────────────────────────────────────────────────────────╯

# Log messages with semantic typing and consistent visual grammar
# Message protocol with strict type enforcement and error isolation
# @usage: log <level> <message>
# @param level: {String} "info"|"success"|"warn"|"error"|"debug"
# @param message: {String} The message content to display
# @returns: {Integer} 0=success (guaranteed execution completion)
log() {
  local -r level="${1:?Missing required log level parameter}"
  local -r message="${2:?Missing required message parameter}"

  # Respect quiet mode with selective override for critical levels
  [[ "${QUIET_MODE:-false}" == "true" && "$level" != "error" && "$level" != "fatal" ]] && return 0

  # Type-safe message rendering with pipe-aware output routing
  case "$level" in
    info) echo -e "$ICON_INFO $message" ;;
    success) echo -e "$ICON_SUCCESS $message" ;;
    warn) echo -e "$ICON_WARNING $message" ;;
    error) echo -e "$ICON_ERROR $message" >&2 ;; # Errors route to stderr
    debug) [[ "${VERBOSE_MODE:-false}" == "true" ]] && echo -e "$ICON_DEBUG $STYLE_DIM$message$STYLE_RESET" ;;
    fatal)
      echo -e "$ICON_ERROR $STYLE_BOLD$message$STYLE_RESET" >&2
      exit 1
      ;;                                        # Fatal terminates execution
    query) echo -e "$ICON_QUESTION $message" ;; # Interactive prompts
    *) echo "$message" ;;                       # Default fallback with no formatting (graceful degradation)
  esac

  return 0 # Type-safe guaranteed return value
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ USER INTERFACE - INFORMATION ARCHITECTURE                            │
# ╰──────────────────────────────────────────────────────────────────────╯

# Knowledge transmission with optimal information density
# Usage: show_help
# Returns: None (void function with side effect of displaying help text)
show_help() {
  cat << EOF
${STYLE_BOLD}EIDOSIAN FORGE REPOSITORY RESTRUCTURER${STYLE_NORMAL}

Transforms chaotic repositories into structured poetry with atomic precision.

${STYLE_BOLD}USAGE:${STYLE_NORMAL}
    $(basename "$0") [OPTIONS]

${STYLE_BOLD}OPTIONS:${STYLE_NORMAL}
    -h, --help             Display this help (you're reading it now)
    -r, --repo DIR         Source repository directory (default: current)
    -t, --target DIR       Target directory for restructured project
    -q, --quiet            Operate in stealth mode (minimal output)
    -y, --yes              Assume "yes" to all prompts (confident or reckless)
    -d, --dry-run          Simulate restructuring without changing files
    -v, --verbose          Explain what's happening in excruciating detail

${STYLE_BOLD}EXAMPLES:${STYLE_NORMAL}
    # Interactive restructuring of current directory
    $(basename "$0")

    # Restructure ~/messy-project into ~/clean-project
    $(basename "$0") --repo ~/messy-project --target ~/clean-project

${STYLE_BOLD}NOTE:${STYLE_NORMAL}
    This script doesn't just move files; it imposes order on chaos.
    Always backup before proceeding—archaeology is best left to historians.
EOF
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ CAPABILITY REPORTING - RUNTIME ENVIRONMENT INTROSPECTION             │
# ╰──────────────────────────────────────────────────────────────────────╯

# Display detected terminal capabilities with visual status symbols
# Useful for debugging environment-specific rendering issues
# Usage: report_capabilities
# Returns: 0 always (guaranteed execution completion)
report_capabilities() {
  # Respect quiet mode setting - exit early if quiet
  [[ "$QUIET_MODE" == "true" ]] && return 0

  log "info" "╭────────────────────────────────────────────────╮"
  log "info" "│  🔍 DETECTED CAPABILITIES                      │"
  log "info" "╰────────────────────────────────────────────────╯"

  # Transform boolean strings into visual status indicators
  local status_enabled="$(${TERMINAL_SUPPORTS_UNICODE} && echo '✅ Enabled' || echo 'Enabled')"
  local status_disabled="$(${TERMINAL_SUPPORTS_UNICODE} && echo '❌ Disabled' || echo 'Disabled')"

  # Report each capability with consistent formatting
  for feature in "${FEATURES[@]}"; do
    local name="${feature%%:*}"
    local value="${feature#*:}"

    local status="$status_disabled"
    [[ "$value" == "true" ]] && status="$status_enabled"

    log "info" "  ${STYLE_BOLD}${name}${STYLE_RESET}: $status"
  done

  # Environment-specific insights for improved user context
  if [[ "$TERMINAL_SUPPORTS_COLOR" != "true" ]]; then
    log "debug" "Color rendering disabled - falling back to plain text"
  fi

  if [[ "$TERMINAL_SUPPORTS_UNICODE" != "true" ]]; then
    log "debug" "Unicode rendering disabled - using ASCII alternatives"
  fi

  log "info" "" # Visual separation for readability
  return 0
}

# ┌────────────────────────────────────────────────────────────┐
# │ USER INTERACTION - BINARY DECISION PROTOCOL                │
# └────────────────────────────────────────────────────────────┘

# Ask user for confirmation unless auto-yes is enabled
# Usage: confirm <prompt> [default]
# Returns: 0 for yes, 1 for no (POSIX-compliant return codes)
confirm() {
  local prompt="$1"
  local default="${2:-n}" # Default to 'no' if not specified (safe default)

  # Bypass confirmation if auto-yes is enabled (non-interactive mode)
  [[ "$AUTO_YES" == "true" ]] && return 0

  # Prepare choices display based on default (visual clarity)
  local choices
  if [[ "$default" == "y" ]]; then
    choices="[Y/n]"
  else
    choices="[y/N]"
  fi

  # Get user input with appropriate prompt (clear intention)
  read -r -p "$prompt $choices: " response
  response=${response,,} # Convert to lowercase for normalization

  # Use default if no input provided (respect default selection)
  if [[ -z "$response" ]]; then
    response="$default"
  fi

  # Return success (0) for yes, failure (1) for no (semantic return codes)
  [[ "$response" == "y" || "$response" == "yes" ]]
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ FILESYSTEM OPERATIONS - STRUCTURAL INTEGRITY                         │
# ╰──────────────────────────────────────────────────────────────────────╯

# Ensure a directory exists or create it with proper error handling
# Usage: ensure_directory <directory_path>
# Returns: 0 on success, 1 on failure (POSIX-compliant)
ensure_directory() {
  local dir="$1"

  # Only attempt creation if directory doesn't exist (idempotent operation)
  if [[ ! -d "$dir" ]]; then
    log "info" "Creating directory: $dir"
    mkdir -p "$dir" || {
      log "error" "Failed to create directory: $dir"
      return 1 # Explicit failure code for error handling
    }
  fi

  return 0 # Explicit success code for clarity
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ PATH PROCESSING - CANONICAL REPRESENTATION                           │
# ╰──────────────────────────────────────────────────────────────────────╯

# Normalize paths to absolute paths with proper expansion
# Usage: normalized_path=$(normalize_path <path>)
# Returns: Canonical absolute path string
normalize_path() {
  local path="$1"

  # Home directory expansion (platform-independent user directory)
  if [[ "$path" == "~"* ]]; then
    path="${HOME}${path:1}"
  fi

  # Relative to absolute path conversion (context independence)
  if [[ ! "$path" == /* ]]; then
    path="$(pwd)/$path"
  fi

  # Clean up path components (handling symlinks, ./, ../, etc.)
  # With graceful fallback if readlink fails
  local clean_path
  clean_path=$(readlink -f "$path" 2> /dev/null || echo "$path")

  echo "$clean_path" # Pure function output
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ REPOSITORY VALIDATION - EXISTENCE & CONTENT VERIFICATION             │
# ╰──────────────────────────────────────────────────────────────────────╯

# Validate source repository existence and content
# Usage: validate_repo <repository_directory>
# Returns: 0 on success, 1 on failure (POSIX-compliant)
validate_repo() {
  local repo_dir="$1"

  # Check existence - fail early principle
  if [[ ! -d "$repo_dir" ]]; then
    log "error" "Repository directory does not exist: $repo_dir"
    return 1
  fi

  # Check content - empty repositories might indicate user error
  if [[ -z "$(ls -A "$repo_dir")" ]]; then
    log "warn" "Repository directory is empty: $repo_dir"
    if ! confirm "Proceed with empty directory?"; then
      return 1
    fi
  fi

  return 0
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ FILE GENERATION - DETERMINISTIC CONTENT CREATION                     │
# ╰──────────────────────────────────────────────────────────────────────╯

# Generate standard project files with collision detection
# Usage: create_project_file <target_dir> <filename> <content>
# Returns: 0 on success, 1 on failure (POSIX-compliant)
create_project_file() {
  local target_dir="$1"
  local filename="$2"
  local content="$3"

  local file_path="$target_dir/$filename"

  # Don't overwrite existing files unless confirmed (data preservation principle)
  if [[ -f "$file_path" ]]; then
    if ! confirm "File already exists: $filename. Overwrite?"; then
      log "info" "Skipping creation of $filename"
      return 0
    fi
  fi

  log "info" "Creating $filename"
  echo "$content" > "$file_path" || {
    log "error" "Failed to create $filename"
    return 1
  }
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ PROJECT SCAFFOLDING - STRUCTURAL TEMPLATE GENERATION                 │
# ╰──────────────────────────────────────────────────────────────────────╯

# Generate standard project structure with directory creation
# Usage: create_project_structure <target_directory>
# Returns: Implicit through function calls
create_project_structure() {
  local target_dir="$1"
  local pkg_name="$(basename "$target_dir" | tr '-' '_')"

  log "info" "Creating standard directory structure..."
  local dirs=(
    "src/$pkg_name"     # Package source
    "tests"             # Test artifacts
    "docs"              # Documentation
    ".github/workflows" # CI pipeline definition
  )

  # Create each directory with validation
  for dir in "${dirs[@]}"; do
    ensure_directory "$target_dir/$dir"
  done

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ PROJECT MANIFESTS - DECLARATIVE IDENTITY                             │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Create pyproject.toml - central configuration nexus
  create_project_file "$target_dir" "pyproject.toml" "$(
    cat << EOF
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "$pkg_name"
version = "0.1.0"
description = "An Eidosian project with structural elegance"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}

[project.scripts]
$pkg_name = "$pkg_name.cli:main"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ BACKWARD COMPATIBILITY - TEMPORAL BRIDGING                           │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Create setup.py - legacy support layer
  create_project_file "$target_dir" "setup.py" "$(
    cat << EOF
#!/usr/bin/env python3
from setuptools import setup

# Using setup.py for compatibility, but prefer pyproject.toml
setup(
        name="$pkg_name",
        package_dir={"": "src"},
        packages=["$pkg_name"],
)
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ PROJECT DOCUMENTATION - KNOWLEDGE TRANSFER PROTOCOL                   │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Create README.md - first contact experience design
  create_project_file "$target_dir" "README.md" "$(
    cat << EOF
# ${pkg_name^} 🔨

> _"Crafted with Eidosian precision"_

## 🚀 Features

- 🧠 Feature one with surgical precision
- 🧩 Feature two that catches issues before they bite
- 🗺️ Feature three that actually makes sense
- ⚡ Feature four with crystal clarity

## 📦 Installation

\`\`\`bash
pip install $pkg_name
\`\`\`

## 🔧 Usage

\`\`\`python
import $pkg_name

# Your elegant code here
\`\`\`

## 🛠️ Development

\`\`\`bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
\`\`\`
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ VERSION CONTROL HYGIENE - ARTIFACT FILTRATION                        │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Create .gitignore - entropy boundary specification
  create_project_file "$target_dir" ".gitignore" "$(
    cat << 'EOF'
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/

# Editors
.idea/
.vscode/
*.swp
*.swo
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ TEST SCAFFOLDING - VERIFICATION ARCHITECTURE                          │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Create foundational test structure
  ensure_directory "$target_dir/tests"
  create_project_file "$target_dir/tests" "__init__.py" '"""Test suite - structural validation nexus."""'
  create_project_file "$target_dir/tests" "test_basic.py" "$(
    cat << EOF
"""Basic tests for $pkg_name - reality validation protocol."""
import pytest

def test_truth():
    """Truth is invariant across runtime contexts."""
    assert True  # Foundational sanity check - universe integrity verification
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ LICENSE PROTOCOL - LEGAL BOUNDARIES                                   │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Establish legal operating parameters
  create_project_file "$target_dir" "LICENSE" "$(
    cat << 'EOF'
MIT License

Copyright (c) 2025 The Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
  )"

  # ╭──────────────────────────────────────────────────────────────────────╮
  # │ CONTINUOUS INTEGRATION - VERIFICATION AUTOMATION                      │
  # ╰──────────────────────────────────────────────────────────────────────╯

  # Define self-validating pipeline with deterministic outcomes
  ensure_directory "$target_dir/.github/workflows"
  create_project_file "$target_dir/.github/workflows" "python-tests.yml" "$(
    cat << 'EOF'
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov black isort
          pip install -e .
      - name: Lint with black
        run: black --check src tests
      - name: Test with pytest
        run: pytest --cov
EOF
  )"
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ FILE MIGRATION - STRUCTURAL TRANSFORMATION                            │
# ╰──────────────────────────────────────────────────────────────────────╯

# Transform source architecture into target architecture with deterministic preservation
move_files() {
  local src_dir="$1"
  local target_dir="$2"

  log "info" "Migrating files with structural preservation..."

  # Catalog existing artifacts with atomic isolation
  local files=$(find "$src_dir" -maxdepth 1 -mindepth 1)

  # Initialize package namespace with proper identity
  local pkg_name="$(basename "$target_dir" | tr '-' '_')"
  local pkg_dir="$target_dir/src/$pkg_name"
  ensure_directory "$pkg_dir"

  # Define structural boundaries with explicit exclusion parameters
  local excluded_patterns=(
    "^src$"
    "^tests$"
    "^docs$"
    "^\.github$"
    "^__pycache__$"
    "^\.git$"
  )

  # Process each artifact with individual transformation protocol
  for src_path in $files; do
    local file="$(basename "$src_path")"
    local excluded=0

    # Apply exclusion rules with deterministic pattern matching
    for pattern in "${excluded_patterns[@]}"; do
      if [[ "$file" =~ $pattern ]]; then
        excluded=1
        log "debug" "Excluding $file (matches pattern: $pattern)"
        break
      fi
    done

    # Skip artifacts excluded by structural rules
    if [[ $excluded -eq 1 ]]; then
      log "info" "Skipping $file (excluded by pattern)"
      continue
    fi

    # Prevent recursive duplication with self-reference detection
    if [[ "$src_path" == "$target_dir" ]]; then
      log "debug" "Skipping $file (self-reference protection)"
      continue
    fi

    if [[ -e "$src_path" ]]; then
      log "info" "Migrating $file to package namespace"

      # Perform atomic copy with error detection
      cp -r "$src_path" "$pkg_dir/" || {
        log "error" "Failed to copy $file to $pkg_dir/"
        continue
      }

      # Cleanup source only when paths differ (idempotent operation)
      if [[ "$src_dir" != "$target_dir" && "$AUTO_YES" == "true" ]]; then
        rm -rf "$src_path" || {
          log "warn" "Failed to remove $file from source directory"
        }
      elif [[ "$src_dir" != "$target_dir" ]]; then
        if confirm "Remove original file: $file?" "n"; then
          rm -rf "$src_path" || {
            log "warn" "Failed to remove $file from source directory"
          }
        fi
      fi
    fi
  done

  # Ensure namespace initialization with package marker
  if [[ ! -f "$pkg_dir/__init__.py" ]]; then
    log "info" "Creating package namespace initialization"
    create_project_file "$pkg_dir" "__init__.py" \
      "\"\"\"$pkg_name package - functional nucleus with recursive precision.\"\"\""
  fi

  log "success" "Migration complete - structure transformed with precision"
}

# ╭──────────────────────────────────────────────────────────────────────╮
# │ MAIN EXECUTION PROTOCOL - ORCHESTRATION LOGIC                        │
# ╰──────────────────────────────────────────────────────────────────────╯

main() {
  # ┌────────────────────────────────────────────────────────────┐
  # │ STATE INITIALIZATION - CONFIGURABLE DEFAULTS               │
  # └────────────────────────────────────────────────────────────┘

  local REPO_DIR=$(pwd)
  local TARGET_DIR=$(pwd)
  local QUIET_MODE="false"
  local AUTO_YES="false"
  local DRY_RUN="false"
  local VERBOSE_MODE="false"

  # ┌────────────────────────────────────────────────────────────┐
  # │ ARGUMENT PARSING - DETERMINISTIC INTENT CAPTURE            │
  # └────────────────────────────────────────────────────────────┘

  while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
      -h | --help)
        show_help
        exit 0
        ;;
      -r | --repo)
        [[ -z "$2" || "$2" == -* ]] && {
          log "error" "Missing value for $key"
          exit 1
        }
        REPO_DIR="$2"
        shift 2
        ;;
      -t | --target)
        [[ -z "$2" || "$2" == -* ]] && {
          log "error" "Missing value for $key"
          exit 1
        }
        TARGET_DIR="$2"
        shift 2
        ;;
      -q | --quiet)
        QUIET_MODE="true"
        shift
        ;;
      -y | --yes)
        AUTO_YES="true"
        shift
        ;;
      -d | --dry-run)
        DRY_RUN="true"
        log "info" "Dry run mode active - simulation only"
        shift
        ;;
      -v | --verbose)
        VERBOSE_MODE="true"
        shift
        ;;
      *)
        log "error" "Unknown option: $key"
        show_help
        exit 1
        ;;
    esac
  done

  # ┌────────────────────────────────────────────────────────────┐
  # │ PATH NORMALIZATION - CANONICAL REPRESENTATION              │
  # └────────────────────────────────────────────────────────────┘

  REPO_DIR=$(normalize_path "$REPO_DIR")
  TARGET_DIR=$(normalize_path "$TARGET_DIR")

  # ┌────────────────────────────────────────────────────────────┐
  # │ EXECUTION INITIATION - STRUCTURAL TRANSFORMATION BEGINS    │
  # └────────────────────────────────────────────────────────────┘

  log "info" "╭──────────────────────────────────────────────────╮"
  log "info" "│  🏗️  EIDOSIAN FORGE REPOSITORY RESTRUCTURER      │"
  log "info" "╰──────────────────────────────────────────────────╯"
  log "info" "Source repository: $REPO_DIR"
  log "info" "Target directory: $TARGET_DIR"

  # Mode notification for user orientation
  [[ "$DRY_RUN" == "true" ]] && log "info" "Mode: Simulation (no files will be modified)"

  # ┌────────────────────────────────────────────────────────────┐
  # │ SOURCE VALIDATION - ORIGIN INTEGRITY CHECK                 │
  # └────────────────────────────────────────────────────────────┘

  validate_repo "$REPO_DIR" || {
    log "error" "Invalid repository directory. Exiting with structural integrity."
    exit 1
  }

  # ┌────────────────────────────────────────────────────────────┐
  # │ TARGET VALIDATION - DESTINATION INTEGRITY CHECK            │
  # └────────────────────────────────────────────────────────────┘

  if [[ ! -d "$TARGET_DIR" ]]; then
    if confirm "Target directory doesn't exist. Create it?" "y"; then
      if [[ "$DRY_RUN" != "true" ]]; then
        ensure_directory "$TARGET_DIR" || {
          log "error" "Failed to create target directory. Exiting."
          exit 1
        }
      else
        log "debug" "Would create directory: $TARGET_DIR"
      fi
    else
      log "error" "Target directory is required. Exiting."
      exit 1
    fi
  fi

  # ┌────────────────────────────────────────────────────────────┐
  # │ INTENT CONFIRMATION - EXPLICIT USER APPROVAL               │
  # └────────────────────────────────────────────────────────────┘

  if ! confirm "Ready to restructure the repository. Proceed?" "y"; then
    log "info" "Operation canceled by user. Exiting with no changes."
    exit 0
  fi

  # ┌────────────────────────────────────────────────────────────┐
  # │ TRANSFORMATION EXECUTION - ATOMIC STATE CHANGE             │
  # └────────────────────────────────────────────────────────────┘

  if [[ "$DRY_RUN" != "true" ]]; then
    # Structure before content - establish architectural foundation
    create_project_structure "$TARGET_DIR"

    # Content migration with structural preservation
    move_files "$REPO_DIR" "$TARGET_DIR"
  else
    log "info" "Dry run complete - transformation simulated without modification"
    exit 0
  fi

  # ┌────────────────────────────────────────────────────────────┐
  # │ OPERATION COMPLETION - GUIDANCE PROTOCOL                   │
  # └────────────────────────────────────────────────────────────┘

  log "success" "✨ Repository restructuring complete! ✨"
  log "info" "Your project now follows crystalline structural conventions."
  log "info" ""
  log "info" "🧪 To initialize git (if not already done):"
  log "info" "cd \"$TARGET_DIR\""
  log "info" "git init"
  log "info" "git add ."
  log "info" "git commit -m \"🏗️ Initial project structure with Eidosian precision\""
  log "info" ""
  log "info" "💻 To install your package in development mode:"
  log "info" "cd \"$TARGET_DIR\""
  log "info" "pip install -e ."
}

# Execute main function with argument propagation
main "$@"
