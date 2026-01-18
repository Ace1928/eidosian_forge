#!/usr/bin/env bash
# 🧪 Eidosian universal test runner
# Runs tests for all projects in the monorepo

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECTS_DIR="$REPO_ROOT/projects"
LOG_FILE="$REPO_ROOT/test_log.txt"

echo "🔮 Eidosian Universal Test Runner"
echo "================================"
echo "Running tests for all projects..."
echo "" > "$LOG_FILE"

# Function to test Python projects
test_python_project() {
    local project_dir="$1"
    echo "🐍 Testing Python project: $(basename "$project_dir")"
    
    if [ -d "$project_dir/tests" ]; then
        (cd "$project_dir" && python -m pytest) || echo "⚠️  Tests failed for $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No tests directory found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to test Node.js projects
test_nodejs_project() {
    local project_dir="$1"
    echo "🟢 Testing Node.js project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/package.json" ]; then
        (cd "$project_dir" && npm test) || echo "⚠️  Tests failed for $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No package.json found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to test Go projects
test_go_project() {
    local project_dir="$1"
    echo "🔵 Testing Go project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/go.mod" ]; then
        (cd "$project_dir" && go test ./...) || echo "⚠️  Tests failed for $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No go.mod found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to test Rust projects
test_rust_project() {
    local project_dir="$1"
    echo "🦀 Testing Rust project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/Cargo.toml" ]; then
        (cd "$project_dir" && cargo test) || echo "⚠️  Tests failed for $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No Cargo.toml found in $(basename "$project_dir"). Skipping."
    fi
}

# Find and test all projects
if [ -d "$PROJECTS_DIR" ]; then
    for project_dir in "$PROJECTS_DIR"/*; do
        if [ -d "$project_dir" ]; then
            if [[ "$project_dir" == *python_project ]]; then
                test_python_project "$project_dir"
            elif [[ "$project_dir" == *nodejs_project ]]; then
                test_nodejs_project "$project_dir"
            elif [[ "$project_dir" == *go_project ]]; then
                test_go_project "$project_dir"
            elif [[ "$project_dir" == *rust_project ]]; then
                test_rust_project "$project_dir"
            else:
                echo "⚠️ Unknown project type: $(basename "$project_dir"). Skipping."
            fi
        fi
    done
else
    echo "⚠️ Projects directory not found at $PROJECTS_DIR"
fi

echo ""
echo "✨ Test run completed!"
echo "Check $LOG_FILE for any errors."
