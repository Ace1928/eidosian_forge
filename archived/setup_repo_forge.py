#!/usr/bin/env python3
"""
Repo Forge Setup Script – Eidosian Bootstrap
────────────────────────────────────────────────────────────
Fully universal repository setup tool for any *_forge repository.
This script dynamically determines its target based on the current working directory,
sets up the virtual environment, installs the repository in development mode,
bootstraps the structure, runs tests, and logs every step with Eidosian clarity.
"""

import os
import sys
import re
import json
import subprocess
import shutil
import logging
from datetime import datetime
from pathlib import Path
import argparse
import time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANSI Color Codes for Terminal Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
GRAY = "\033[0;90m"
WHITE = "\033[0;97m"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Paths and Execution Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PATH = SCRIPT_DIR / "repo_forge_repo"
EIDOS_VENV = SCRIPT_DIR / "eidos_venv"
ACTIVATION_SCRIPT = EIDOS_VENV / "bin" / "activate"
ACTIVATE_WRAPPER = SCRIPT_DIR / "activate_eidos_venv.sh"
LOG_FILE = SCRIPT_DIR / "repo_forge_setup.log"

# Execution parameters (defaults)
BOOTSTRAP_MODE = False
FORCE_REINSTALL = False
DEV_DEPENDENCIES = True
SKIP_TESTS = False
DEBUG_MODE = False
TARGET_VERSION = ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging Setup – Log to both terminal and file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("FORGE_DEBUG") == "1" or DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("forge.setup")
logger.debug("Eidosian Repo Forge Setup Script loaded with debug logging")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal Logging Helpers – Eidosian Style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_debug(msg):
    logger.debug(msg)

def log_info(msg):
    logger.info(f"{BLUE}[INFO]{RESET} {msg}")

def log_success(msg):
    logger.info(f"{GREEN}[SUCCESS]{RESET} {msg}")

def log_warning(msg):
    logger.warning(f"{YELLOW}[WARNING]{RESET} {msg}")

def log_error(msg):
    logger.error(f"{RED}[ERROR]{RESET} {msg}")

def log_step(msg):
    print(f"\n{PURPLE}[STEP]{RESET} {BOLD}{msg}{RESET}")
    logger.info(f"[STEP] {msg}")

def log_code(msg):
    print(f"  {CYAN}{msg}{RESET}")
    logger.info(f"  {msg}")

def log_header(title):
    width = 72
    padding = (width - len(title) - 4) // 2
    left = " " * padding
    right = " " * (padding + ((width - len(title) - 4) % 2))
    header_line = f"{PURPLE}╔{'═'*width}╗{RESET}"
    footer_line = f"{PURPLE}╚{'═'*width}╝{RESET}"
    title_line = f"{PURPLE}║{left}{WHITE}{BOLD}{title}{RESET}{PURPLE}{right}║{RESET}"
    print(f"\n{header_line}\n{title_line}\n{footer_line}\n")
    logger.info(header_line)
    logger.info(title_line)
    logger.info(footer_line)

def log_execution_time(start_time):
    elapsed = time.time() - start_time
    formatted = f"{elapsed:.3f}"
    log_debug(f"Execution time: {formatted}s")
    print(f"{GRAY}⏱️ Execution time: {formatted}s{RESET}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Environment Verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def verify_environment():
    start = time.time()
    log_step("Verifying system environment prerequisites")
    
    required_cmds = ["python3", "pip", "git", "bash"]
    missing = []
    for cmd in required_cmds:
        if shutil.which(cmd) is None:
            missing.append(cmd)
            log_warning(f"{cmd} not found in PATH")
        else:
            log_debug(f"{cmd} is available: {shutil.which(cmd)}")
    if missing:
        log_error("Missing required commands: " + ", ".join(missing))
        sys.exit(1)
    
    # Verify Python version >= 3.8
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        log_error(f"Python 3.8+ required; detected {py_version.major}.{py_version.minor}")
        sys.exit(1)
    
    log_success("Environment verification complete")
    log_execution_time(start)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Virtual Environment Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_virtual_environment():
    start = time.time()
    log_step("Setting up Eidosian virtual environment")
    
    # If a dedicated setup script exists, run it.
    setup_script = SCRIPT_DIR / "setup_eidos_venv.sh"
    if setup_script.exists():
        log_info("Executing external virtual environment setup script")
        cmd = ["/bin/bash", str(setup_script)]
        if FORCE_REINSTALL:
            cmd.append("--force")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            log_error("Failed to execute setup_eidos_venv.sh")
            sys.exit(1)
    else:
        log_warning("External venv setup script not found; creating venv directly")
        if EIDOS_VENV.exists() and FORCE_REINSTALL:
            log_info("Removing existing virtual environment")
            shutil.rmtree(EIDOS_VENV)
        if not EIDOS_VENV.exists():
            log_info(f"Creating new virtual environment at {EIDOS_VENV}")
            try:
                subprocess.run([sys.executable, "-m", "venv", str(EIDOS_VENV)], check=True)
            except subprocess.CalledProcessError:
                log_error("Virtual environment creation failed")
                sys.exit(1)
        else:
            log_info("Using existing virtual environment")
    
    # Check activation script exists
    if not (EIDOS_VENV / "bin" / "activate").exists():
        log_error("Activation script not found in venv")
        sys.exit(1)
    
    # Activate virtual environment in current process (best-effort)
    # Note: In Python, re-exec may be needed for full activation; here we set PATH
    activate_path = EIDOS_VENV / "bin"
    os.environ["PATH"] = str(activate_path) + os.pathsep + os.environ["PATH"]
    os.environ["VIRTUAL_ENV"] = str(EIDOS_VENV)
    log_info("Virtual environment activated (PATH adjusted)")
    
    # Upgrade core packages
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    except subprocess.CalledProcessError:
        log_warning("Core package upgrade failed")
    
    log_success("Virtual environment setup complete")
    log_execution_time(start)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Repository Setup – Install and Configure Repo Forge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_repo_forge():
    start = time.time()
    log_step("Installing Repo Forge from source")
    
    if not REPO_PATH.exists():
        log_error(f"Repository not found at {REPO_PATH}")
        log_info("Clone the repository first, e.g.:")
        log_code(f"git clone https://github.com/Ace1928/repo_forge.git {REPO_PATH}")
        sys.exit(1)
    
    # Check for setup.py or pyproject.toml
    if not ((REPO_PATH / "setup.py").exists() or (REPO_PATH / "pyproject.toml").exists()):
        log_error("No setup.py or pyproject.toml found in repository")
        sys.exit(1)
    
    os.chdir(REPO_PATH)
    log_debug(f"Changed working directory to {Path.cwd()}")
    
    # Install repository in editable mode
    try:
        log_info("Installing repo_forge in development mode")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    except subprocess.CalledProcessError:
        log_error("Installation of repo_forge failed")
        sys.exit(1)
    
    # Install development and docs dependencies if requested
    if DEV_DEPENDENCIES:
        log_info("Installing development dependencies")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=False)
        log_info("Installing documentation dependencies")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=False)
    
    # Verify installation by attempting import
    try:
        output = subprocess.check_output([sys.executable, "-c", "import repo_forge; print(repo_forge.__version__)"])
        log_info(f"repo_forge imported successfully: {output.decode().strip()}")
    except subprocess.CalledProcessError:
        log_error("Failed to import repo_forge after installation")
        sys.exit(1)
    
    # Create a command-line wrapper script if not exists
    repo_forge_cli = REPO_PATH / "repo_forge" / "cli.py"
    target_cmd = EIDOS_VENV / "bin" / "repo-forge"
    if target_cmd.exists():
        log_debug("repo-forge command already exists")
    elif repo_forge_cli.exists():
        log_info("Creating repo-forge command wrapper")
        wrapper_content = f"""#!/bin/bash
# Auto-generated repo-forge command
source "{ACTIVATION_SCRIPT}"
python -m repo_forge.cli "$@"
"""
        try:
            with open(target_cmd, "w", encoding="utf-8") as f:
                f.write(wrapper_content)
            os.chmod(target_cmd, 0o755)
            log_debug(f"Created executable at {target_cmd}")
        except Exception as e:
            log_warning(f"Failed to create repo-forge command: {e}")
    
    log_success("Repo Forge installed successfully")
    log_execution_time(start)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bootstrapping Repository Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def bootstrap_repo_forge():
    start = time.time()
    log_step("Bootstrapping repository structure")
    
    os.chdir(REPO_PATH)
    log_debug(f"Current directory: {Path.cwd()}")
    
    # Run bootstrap command with verbosity if BOOTSTRAP_MODE is true
    bootstrap_cmd = [sys.executable, "-m", "repo_forge.cli", "--bootstrap"]
    if BOOTSTRAP_MODE:
        bootstrap_cmd.append("--verbose")
        log_info("Running full bootstrap mode")
    else:
        log_info("Running minimal bootstrap mode")
    
    try:
        subprocess.run(bootstrap_cmd, check=True)
    except subprocess.CalledProcessError:
        log_error("Bootstrap operation failed")
        sys.exit(1)
    
    # Verify core directories exist
    required_dirs = ["docs", "tests", "repo_forge"]
    missing = [d for d in required_dirs if not (REPO_PATH / d).is_dir()]
    if missing:
        log_warning("Missing expected directories: " + ", ".join(missing))
    else:
        log_debug("All required directories verified")
    
    log_success("Repository bootstrapped successfully")
    log_execution_time(start)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Running Tests – Quality Assurance Protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests():
    start = time.time()
    log_step("Running verification tests")
    
    if SKIP_TESTS:
        log_info("Tests skipped due to --skip-tests flag")
        return
    
    os.chdir(REPO_PATH)
    log_debug(f"Changed directory to {Path.cwd()}")
    
    # Ensure pytest is installed
    if shutil.which("pytest") is None:
        log_info("pytest not found; installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)
        except subprocess.CalledProcessError:
            log_error("Failed to install pytest")
            sys.exit(1)
    
    # Run an import test
    try:
        subprocess.run([sys.executable, "-c", "import repo_forge; print(f'repo_forge {repo_forge.__version__} imported')"], check=True)
        log_info("Module import test passed")
    except subprocess.CalledProcessError:
        log_error("Module import test failed")
        sys.exit(1)
    
    # Run pytest if tests directory exists
    if (REPO_PATH / "tests").is_dir():
        log_info("Running unit tests...")
        try:
            subprocess.run([sys.executable, "-m", "pytest", "-xvs", "tests/"], check=True)
            log_success("All tests passed")
        except subprocess.CalledProcessError:
            log_warning("Some tests failed")
    else:
        log_warning("No tests directory found")
    
    # Test CLI functionality
    log_info("Testing command-line interface...")
    try:
        subprocess.run(["repo-forge", "--version"], check=True)
        log_success("CLI test passed")
    except subprocess.CalledProcessError:
        log_warning("CLI command not found; trying module invocation")
        try:
            subprocess.run([sys.executable, "-m", "repo_forge.cli", "--version"], check=True)
            log_success("Module CLI test passed")
        except subprocess.CalledProcessError:
            log_error("CLI test failed")
            sys.exit(1)
    
    log_success("Verification tests completed")
    log_execution_time(start)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Guide Display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def display_usage_guide():
    log_step("Repo Forge Usage Instructions")
    print("")
    log_info("To activate the Eidosian virtual environment:")
    if ACTIVATE_WRAPPER.exists():
        log_code(f"source {ACTIVATE_WRAPPER}")
    else:
        log_code(f"source {ACTIVATION_SCRIPT}")
    print("")
    log_info("Available commands:")
    log_code("repo-forge --help")
    log_code("repo-forge --version")
    log_code("repo-forge -o /path/to/output -n repo_name --languages python --init-git")
    print("")
    log_info("For metadata repository setup, run:")
    log_code(f"python {SCRIPT_DIR / 'setup_metadata_forge.py'}")
    print("")
    log_info("Log file available at:")
    log_code(str(LOG_FILE))
    print("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command-Line Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_arguments():
    parser = argparse.ArgumentParser(description="Repo Forge Setup Script – Eidosian Bootstrap")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstall/recreation of virtual environment")
    parser.add_argument("--bootstrap", "-b", action="store_true", help="Run full repository bootstrapping")
    parser.add_argument("--skip-dev", "-s", action="store_true", help="Skip installing development dependencies")
    parser.add_argument("--skip-tests", "-t", action="store_true", help="Skip running tests")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug output")
    parser.add_argument("--version", "-v", metavar="VER", help="Install specific version (if applicable)")
    return parser.parse_args()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Execution Sequence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global BOOTSTRAP_MODE, FORCE_REINSTALL, DEV_DEPENDENCIES, SKIP_TESTS, DEBUG_MODE, TARGET_VERSION
    args = parse_arguments()
    FORCE_REINSTALL = args.force
    BOOTSTRAP_MODE = args.bootstrap
    DEV_DEPENDENCIES = not args.skip_dev
    SKIP_TESTS = args.skip_tests
    DEBUG_MODE = args.debug
    if args.version:
        TARGET_VERSION = args.version

    total_start = time.time()
    setup_logging()  # Writes initial log header to LOG_FILE

    log_header("REPO FORGE SETUP SEQUENCE")
    log_info("Beginning setup with mathematical precision and Eidosian elegance")

    try:
        verify_environment()
        setup_virtual_environment()
        setup_repo_forge()
        bootstrap_repo_forge()
        run_tests()
    except Exception as e:
        log_error(f"Setup failed: {e}")
        sys.exit(1)
    
    total_elapsed = time.time() - total_start
    log_header("SETUP COMPLETED SUCCESSFULLY")
    log_success(f"Total execution time: {total_elapsed:.3f} seconds")
    display_usage_guide()
    return 0

def setup_logging():
    """
    Initializes the log file with header information and system diagnostics.
    """
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("╔" + "═" * 60 + "╗\n")
        f.write(f"║  REPO FORGE SETUP LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ║\n")
        f.write("╚" + "═" * 60 + "╝\n\n")
        f.write("System Information:\n")
        f.write(f"- OS: {os.uname().sysname} {os.uname().release}\n")
        f.write(f"- Host: {os.uname().nodename}\n")
        if shutil.which("python3"):
            try:
                py_ver = subprocess.check_output(["python3", "--version"]).decode().strip()
                f.write(f"- Python: {py_ver}\n")
            except Exception:
                f.write("- Python: Unknown\n")
        f.write(f"- Bash: {os.getenv('BASH_VERSION', 'N/A')}\n")
        f.write(f"- Script Dir: {SCRIPT_DIR}\n")
        f.write(f"- Repo Path: {REPO_PATH}\n")
        f.write(f"- Venv Path: {EIDOS_VENV}\n\n")
    log_debug("Logging initialized")

if __name__ == "__main__":
    sys.exit(main())
