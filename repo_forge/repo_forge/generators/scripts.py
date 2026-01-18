"""
Script generator for Eidosian repositories.

This module generates utility scripts and automation tools for the repository,
following Eidosian principles of functionality and enhancement.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import os
import importlib.util
import sys
from datetime import datetime

from ..core.files import write_file
from ..core.templates import render_template, render_comment_block
from ..core.utils import make_executable


def _load_global_info() -> Dict[str, Any]:
    """
    Load global information from the global_info module.
    
    Gracefully falls back to default values if the module can't be loaded,
    ensuring consistent behavior in any environment.
    
    Returns:
        Dictionary with author information
    """
    try:
        # Try to import global_info module if it's in the path
        global_info_path = Path("/home/lloyd/repos/global_info.py")
        if global_info_path.exists():
            spec = importlib.util.spec_from_file_location("global_info", global_info_path)
            if spec and spec.loader:
                global_info = importlib.util.module_from_spec(spec)
                sys.modules["global_info"] = global_info
                spec.loader.exec_module(global_info)
                logging.debug("Successfully loaded global_info from %s", global_info_path)
                return {
                    "author": global_info.AUTHOR_INFO,
                }
    except Exception as e:
        logging.debug(f"Could not load global_info: {e}")
    
    # Fallback values - streamlined for minimal duplication
    logging.debug("Using fallback values for global_info in scripts generator")
    return {
        "author": {
            "name": "Eidosian Team",
            "email": "team@example.com",
            "org": "Eidosian",
        }
    }


def create_script_files(
    base_path: Path,
    languages: Optional[List[str]] = None,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Create automation script files for the repository.
    
    Args:
        base_path: Base repository directory
        languages: Programming languages to support
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with creation results
    """
    if languages is None:
        languages = ["python", "nodejs", "go", "rust"]
    
    scripts_path = base_path / "scripts"
    scripts_path.mkdir(exist_ok=True, parents=True)
    
    created_files = []
    
    # Create README for scripts directory
    readme_content = """# 🛠️ Automation Scripts

This directory contains scripts for automating common tasks in the repository.

## Directory Structure

- `build/`: Build automation scripts
- `deploy/`: Deployment scripts
- `setup/`: Environment setup scripts
- `maintenance/`: System maintenance scripts
- `database/`: Database management scripts
- `utils/`: Utility scripts
- `ci/`: CI/CD helper scripts
- `dev/`: Development utility scripts

## Usage

Most scripts can be run directly from the command line:

```bash
./scripts/setup/install_dependencies.sh
```

## Contributing

When adding new scripts:
1. Make sure they are executable (`chmod +x script.sh`)
2. Add appropriate documentation and usage examples
3. Follow the repository's coding standards
"""
    write_file(scripts_path / "README.md", readme_content, overwrite)
    created_files.append("scripts/README.md")
    
    # Create setup scripts
    setup_path = scripts_path / "setup"
    setup_path.mkdir(exist_ok=True, parents=True)
    
    # Create install dependencies script
    install_script = """#!/usr/bin/env bash
# 🚀 Eidosian dependency installer
# Automatically installs dependencies for all supported languages

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$REPO_ROOT/setup_log.txt"

echo "🔮 Eidosian Dependency Installer"
echo "================================"
echo "Installing all required dependencies..."

# Check for Python
if command -v python3 &> /dev/null; then
    echo "✅ Python detected. Installing Python dependencies..."
    if [ -f "$REPO_ROOT/requirements.txt" ]; then
        python3 -m pip install -r "$REPO_ROOT/requirements.txt" || { 
            echo "⚠️ Python dependency installation failed with exit code $?" >> "$LOG_FILE"
            echo "⚠️ Python dependency installation failed" 
        }
    else
        echo "⚠️ No requirements.txt found. Skipping Python dependencies."
    fi
else
    echo "⚠️ Python not found. Skipping Python dependencies."
fi

# Check for Node.js
if command -v node &> /dev/null; then
    echo "✅ Node.js detected. Installing JavaScript dependencies..."
    if [ -f "$REPO_ROOT/package.json" ]; then
        (cd "$REPO_ROOT" && npm install) || echo "⚠️  Node.js dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No package.json found. Skipping JavaScript dependencies."
    fi
else
    echo "⚠️  Node.js not found. Skipping JavaScript dependencies."
fi

# Check for Go
if command -v go &> /dev/null; then
    echo "✅ Go detected. Installing Go dependencies..."
    if [ -f "$REPO_ROOT/go.mod" ]; then
        (cd "$REPO_ROOT" && go mod download) || echo "⚠️  Go dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No go.mod found. Skipping Go dependencies."
    fi
else
    echo "⚠️  Go not found. Skipping Go dependencies."
fi

# Check for Rust
if command -v cargo &> /dev/null; then
    echo "✅ Rust detected. Installing Rust dependencies..."
    if [ -f "$REPO_ROOT/Cargo.toml" ]; then
        (cd "$REPO_ROOT" && cargo fetch) || echo "⚠️  Rust dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No Cargo.toml found. Skipping Rust dependencies."
    fi
else
    echo "⚠️  Rust not found. Skipping Rust dependencies."
fi

echo ""
echo "✨ Dependency installation completed!"
echo "Check $LOG_FILE for any errors."
"""
    write_file(setup_path / "install_dependencies.sh", install_script)
    created_files.append("scripts/setup/install_dependencies.sh")
    make_executable(setup_path / "install_dependencies.sh")
    
    # Create build script
    build_path = scripts_path / "build"
    build_path.mkdir(exist_ok=True, parents=True)
    
    build_script = """#!/usr/bin/env bash
# 🏗️ Eidosian universal build script
# Builds all projects in the monorepo

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECTS_DIR="$REPO_ROOT/projects"
LOG_FILE="$REPO_ROOT/build_log.txt"

echo "🔮 Eidosian Universal Builder"
echo "============================"
echo "Building all projects..."
echo "" > "$LOG_FILE"

# Function to build Python projects
build_python_project() {
    local project_dir="$1"
    echo "🐍 Building Python project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/pyproject.toml" ]; then
        (cd "$project_dir" && python -m build) || echo "⚠️  Failed to build $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No pyproject.toml found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to build Node.js projects
build_nodejs_project() {
    local project_dir="$1"
    echo "🟢 Building Node.js project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/package.json" ]; then
        (cd "$project_dir" && npm run build) || echo "⚠️  Failed to build $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No package.json found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to build Go projects
build_go_project() {
    local project_dir="$1"
    echo "🔵 Building Go project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/go.mod" ]; then
        (cd "$project_dir" && go build ./...) || echo "⚠️  Failed to build $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No go.mod found in $(basename "$project_dir"). Skipping."
    fi
}

# Function to build Rust projects
build_rust_project() {
    local project_dir="$1"
    echo "🦀 Building Rust project: $(basename "$project_dir")"
    
    if [ -f "$project_dir/Cargo.toml" ]; then
        (cd "$project_dir" && cargo build) || echo "⚠️  Failed to build $(basename "$project_dir")" >> "$LOG_FILE"
    else
        echo "⚠️  No Cargo.toml found in $(basename "$project_dir"). Skipping."
    fi
}

# Find and build all projects
if [ -d "$PROJECTS_DIR" ]; then
    for project_dir in "$PROJECTS_DIR"/*; do
        if [ -d "$project_dir" ]; then
            if [[ "$project_dir" == *python_project ]]; then
                build_python_project "$project_dir"
            elif [[ "$project_dir" == *nodejs_project ]]; then
                build_nodejs_project "$project_dir"
            elif [[ "$project_dir" == *go_project ]]; then
                build_go_project "$project_dir"
            elif [[ "$project_dir" == *rust_project ]]; then
                build_rust_project "$project_dir"
            else:
                echo "⚠️ Unknown project type: $(basename "$project_dir"). Skipping."
            fi
        fi
    done
else
    echo "⚠️ Projects directory not found at $PROJECTS_DIR"
fi

echo ""
echo "✨ Build process completed!"
echo "Check $LOG_FILE for any errors."
"""
    write_file(build_path / "build_all.sh", build_script)
    created_files.append("scripts/build/build_all.sh")
    make_executable(build_path / "build_all.sh")
    
    # Create test script
    test_path = scripts_path / "dev"
    test_path.mkdir(exist_ok=True, parents=True)
    
    test_script = """#!/usr/bin/env bash
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
"""
    write_file(test_path / "run_tests.sh", test_script)
    created_files.append("scripts/dev/run_tests.sh")
    make_executable(test_path / "run_tests.sh")
    
    # Create Python script for additional functionality
    if "python" in languages:
        utils_path = scripts_path / "utils"
        utils_path.mkdir(exist_ok=True, parents=True)
        
        global_info = _load_global_info()
        
        project_stats_script = f'''#!/usr/bin/env python3
# 📊 Eidosian Project Stats Generator
# Analyzes projects and generates stats about the codebase
#
# Author: {global_info["author"]["name"]} <{global_info["author"]["email"]}>
# Organization: {global_info["author"]["org"]}
# Created: {datetime.now().strftime("%Y-%m-%d")}

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

def count_lines_by_extension(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception:
        return 0

def get_file_stats(repo_path):
    """Get statistics about files in the repository."""
    # Escaped literal dictionary inside lambda
    stats = defaultdict(lambda: {{"count": 0, "lines": 0}})
    total_files = 0
    total_lines = 0
    
    for root, _, files in os.walk(repo_path):
        # Skip hidden directories and node_modules
        if "/." in root or "node_modules" in root or "target" in root or "__pycache__" in root:
            continue
            
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lstrip('.').lower() or 'no_extension'
            
            lines = count_lines_by_extension(file_path)
            
            stats[ext]["count"] += 1
            stats[ext]["lines"] += lines
            total_files += 1
            total_lines += lines
    
    # Escaped literal dictionary for return
    return {{
        "by_extension": dict(stats),
        "total_files": total_files,
        "total_lines": total_lines,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }}

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        # Default to the repository root
        script_dir = Path(__file__).parent.absolute()
        repo_path = script_dir.parent.parent
    
    # The following f-strings are meant to be preserved in the generated file,
    # so we escape their curly braces.
    print(f"🔮 Eidosian Project Stats Generator")
    print(f"==================================")
    print(f"Analyzing repository: {{repo_path}}")
    
    stats = get_file_stats(repo_path)
    
    # Print summary
    print(f"\\nRepository Summary:")
    print(f"-------------------")
    print(f"Total files: {{stats['total_files']}}")
    print(f"Total lines: {{stats['total_lines']:,}}")
    print("\\nFiles by extension:")
    
    # Sort extensions by line count
    sorted_extensions = sorted(
        stats["by_extension"].items(),
        key=lambda x: x[1]["lines"],
        reverse=True
    )
    
    for ext, data in sorted_extensions:
        print(f"  .{{ext:<10}} {{data['count']:5}} files, {{data['lines']:8,}} lines")
    
    # Save to JSON
    output_path = os.path.join(repo_path, "project_stats.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\\n✨ Stats saved to {{output_path}}")
    print(f"Script stats: {{stats}}")  # Clean, direct logging without complex formatters

if __name__ == "__main__":
    main()
'''
    
        write_file(utils_path / "project_stats.py", project_stats_script, overwrite)
        created_files.append("scripts/utils/project_stats.py")
        make_executable(utils_path / "project_stats.py")

    logging.info(f"Created {len(created_files)} script files")

    # Atomic, optimized return dictionary - zero waste, maximum clarity
    # Force count to integer to ensure JSON serialization stability
    return {
        "success": True,
        "created_files": [str(f) for f in created_files],
        "count": int(len(created_files)),  # Explicit int cast for stability
        "base_path": str(base_path)
    }
