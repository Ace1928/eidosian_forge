#!/usr/bin/env bash
# 🔍 Eidosian Local CI/CD Runner
# Run all CI/CD checks locally before pushing
# Usage: ./scripts/run-checks.sh [--fix] [--python-only] [--ts-only]

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
FIX_MODE=false
PYTHON_ONLY=false
TS_ONLY=false

# ═══════════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
  case $1 in
    --fix)
      FIX_MODE=true
      shift
      ;;
    --python-only)
      PYTHON_ONLY=true
      shift
      ;;
    --ts-only)
      TS_ONLY=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --fix          Auto-fix issues where possible"
      echo "  --python-only  Run only Python checks"
      echo "  --ts-only      Run only TypeScript/JavaScript checks"
      echo "  --help, -h     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
  echo -e "${RED}[✗]${NC} $1"
}

section() {
  echo ""
  echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
  echo ""
}

run_check() {
  local name="$1"
  local command="$2"
  local required="${3:-false}"
  
  log_info "Running: $name"
  
  if eval "$command"; then
    log_success "$name passed"
    return 0
  else
    if [ "$required" = "true" ]; then
      log_error "$name failed (required)"
      return 1
    else
      log_warning "$name failed (optional)"
      return 0
    fi
  fi
}

# ═══════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════

FAILED=0

if [ "$TS_ONLY" = "false" ]; then
  section "🐍 PYTHON CHECKS"
  
  # Check if tools are installed
  log_info "Checking Python tools..."
  if ! command -v black >/dev/null 2>&1; then
    log_warning "black not installed, installing..."
    pip install black isort ruff flake8 mypy || { log_error "Failed to install Python tools"; exit 1; }
  fi
  
  if [ "$FIX_MODE" = "true" ]; then
    log_info "Running formatters in fix mode..."
    run_check "Black format" "black . --line-length=120 --exclude='/(\.git|\.venv|venv|eidosian_venv|node_modules|\.cache|dist|build)/'" false || ((FAILED++))
    run_check "isort" "isort . --profile=black --line-length=120 --skip-gitignore" false || ((FAILED++))
    run_check "Ruff autofix" "ruff check . --fix --config=ruff.toml" false || ((FAILED++))
  else
    log_info "Running formatters in check mode..."
    run_check "Black format check" "black . --check --diff --line-length=120 --exclude='/(\.git|\.venv|venv|eidosian_venv|node_modules|\.cache|dist|build)/'" true || ((FAILED++))
    run_check "isort check" "isort . --check --diff --profile=black --line-length=120 --skip-gitignore" true || ((FAILED++))
  fi
  
  log_info "Running linters..."
  run_check "Ruff lint" "ruff check . --config=ruff.toml" true || ((FAILED++))
  run_check "Flake8" "flake8 . --max-line-length=120 --exclude=.git,.venv,venv,eidosian_venv,node_modules,.cache,dist,build,__pycache__ --exit-zero" false
  run_check "Mypy" "mypy . --ignore-missing-imports --show-error-codes --exclude='(\.venv|venv|eidosian_venv|node_modules|\.cache|dist|build)' --no-error-summary" false
  
  if command -v pytest >/dev/null 2>&1; then
    log_info "Running tests..."
    run_check "Pytest" "pytest -v --maxfail=5" false || ((FAILED++))
  else
    log_warning "pytest not installed, skipping tests"
  fi
fi

if [ "$PYTHON_ONLY" = "false" ]; then
  section "📘 TYPESCRIPT/JAVASCRIPT CHECKS"
  
  # Check if Prettier is installed
  if ! command -v prettier >/dev/null 2>&1; then
    log_info "Installing Prettier globally..."
    npm install -g prettier@3.4.2 || log_warning "Failed to install Prettier globally, will use npx"
  fi
  
  if [ "$FIX_MODE" = "true" ]; then
    log_info "Running Prettier in fix mode..."
    if command -v prettier >/dev/null 2>&1; then
      run_check "Prettier format" "prettier '**/*.{ts,tsx,js,jsx,json}' --write --ignore-path .gitignore --ignore-unknown --log-level warn" false || ((FAILED++))
    else
      run_check "Prettier format (npx)" "npx prettier@3.4.2 '**/*.{ts,tsx,js,jsx,json}' --write --ignore-path .gitignore --ignore-unknown --log-level warn" false || ((FAILED++))
    fi
  else
    log_info "Running Prettier in check mode..."
    if command -v prettier >/dev/null 2>&1; then
      run_check "Prettier check" "prettier '**/*.{ts,tsx,js,jsx,json}' --check --ignore-path .gitignore --ignore-unknown --log-level warn" true || ((FAILED++))
    else
      run_check "Prettier check (npx)" "npx prettier@3.4.2 '**/*.{ts,tsx,js,jsx,json}' --check --ignore-path .gitignore --ignore-unknown --log-level warn" true || ((FAILED++))
    fi
  fi
  
  # Check autoseed project
  if [ -d "game_forge/src/autoseed" ]; then
    log_info "Checking autoseed project..."
    cd game_forge/src/autoseed
    
    if [ ! -d "node_modules" ]; then
      log_info "Installing dependencies..."
      npm install || { log_error "npm install failed"; cd "$REPO_ROOT"; exit 1; }
    fi
    
    run_check "ESLint (autoseed)" "npm run lint" false || ((FAILED++))
    run_check "TypeScript check (autoseed)" "npm run typecheck" false || ((FAILED++))
    run_check "Tests (autoseed)" "npm test" false || ((FAILED++))
    
    cd "$REPO_ROOT"
  else
    log_info "autoseed project not found, skipping TypeScript tests"
  fi
fi

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

section "📊 SUMMARY"

if [ $FAILED -eq 0 ]; then
  log_success "All checks passed! ✨"
  echo ""
  log_info "Your code is ready to push!"
  exit 0
else
  log_error "$FAILED check(s) failed"
  echo ""
  if [ "$FIX_MODE" = "false" ]; then
    log_info "Run with --fix to automatically fix formatting issues:"
    echo "  $0 --fix"
  else
    log_info "Some issues need manual fixing. Check the errors above."
  fi
  exit 1
fi
