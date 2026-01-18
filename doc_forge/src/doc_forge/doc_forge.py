#!/usr/bin/env python3
# 🌀 Eidosian Documentation Command Center
"""
Doc Forge - Universal Documentation Command System

A centralized command interface for documentation operations, embodying
Eidosian principles of structure, flow, precision, and self-awareness.
This script orchestrates all documentation processes with efficiency
and elegant control.

Each command is a precision instrument, each workflow a masterpiece of clarity.
"""

import sys
import time
import argparse
import logging
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Import path utilities for perfect path handling
from .utils.paths import get_repo_root, get_docs_dir, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("doc_forge")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛠️ Core paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPO_ROOT = get_repo_root()
DOCS_DIR = get_docs_dir()
BUILD_DIR = DOCS_DIR / "_build"
SCRIPTS_DIR = REPO_ROOT / "scripts"

logger.debug(f"🔍 REPO_ROOT set to: {REPO_ROOT}")
logger.debug(f"🔍 DOCS_DIR set to: {DOCS_DIR}")
logger.debug(f"🔍 BUILD_DIR set to: {BUILD_DIR}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎭 Command execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_command(command: Union[List[str], str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    start_time = time.time()
    process_cwd = cwd or REPO_ROOT

    if isinstance(command, str):
        command = shlex.split(command)

    try:
        process = subprocess.Popen(
            command,
            cwd=process_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        execution_time = time.time() - start_time
        logger.debug(f"Command completed in {execution_time:.2f}s with code {process.returncode}")

        if process.returncode != 0:
            logger.warning(f"Command exited with non-zero code: {process.returncode}")
            if stderr:
                logger.debug(f"stderr: {stderr[:500]}...")

        return process.returncode, stdout, stderr
    except Exception as e:
        logger.error(f"Command execution failed: {command}, Error: {e}")
        return 1, "", str(e)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 Command implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_setup(args: argparse.Namespace) -> int:
    logger.info("🏗️ Setting up documentation environment")
    requirements_path = DOCS_DIR / "requirements.txt"

    if not requirements_path.exists():
        alt_paths = [
            REPO_ROOT / "requirements.txt",
            REPO_ROOT / "requirements" / "docs.txt",
        ]
        for path in alt_paths:
            if path.exists():
                requirements_path = path
                logger.info(f"📄 Using requirements from: {requirements_path}")
                break

    if not requirements_path.exists():
        logger.warning("⚠️ No requirements file found. Creating minimal one.")
        ensure_dir(requirements_path.parent)
        with open(requirements_path, "w") as f:
            f.write("# Documentation dependencies\nsphinx>=4.0.0\nsphinx-rtd-theme>=1.0.0\n")

    logger.info("📦 Installing Python dependencies")
    code, _, err = run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    if code != 0:
        logger.error(f"❌ Failed to install dependencies: {err}")
        return code

    logger.info("📂 Creating directory structure")
    code, _, err = run_command(["chmod", "+x", str(SCRIPTS_DIR / "create_missing_files.sh")])
    if code == 0:
        code, _, err = run_command([str(SCRIPTS_DIR / "create_missing_files.sh")])

    if code != 0:
        logger.error(f"❌ Failed to create directory structure: {err}")
        return code

    BUILD_DIR.mkdir(exist_ok=True, parents=True)
    (BUILD_DIR / "html").mkdir(exist_ok=True)

    logger.info("✅ Documentation environment setup complete")
    return 0

def cmd_build(args: argparse.Namespace) -> int:
    formats = getattr(args, 'formats', None) or ["html"]
    fix = getattr(args, 'fix', False)
    open_after = getattr(args, 'open', False)

    if fix:
        logger.info("🔧 Fixing documentation issues")
        logger.info("🔗 Fixing cross-references")
        code, _, err = run_command([sys.executable, str(SCRIPTS_DIR / "update_cross_references.py"), str(DOCS_DIR)])
        if code != 0:
            logger.warning(f"⚠️ Cross-reference fixing had issues: {err}")

        logger.info("🏝️ Adding orphan directives to standalone files")
        code, _, err = run_command([sys.executable, str(SCRIPTS_DIR / "update_orphan_directives.py"), str(DOCS_DIR)])
        if code != 0:
            logger.warning(f"⚠️ Orphan directive addition had issues: {err}")

    for output_format in formats:
        logger.info(f"📚 Building {output_format.upper()} documentation")
        build_dir = BUILD_DIR / output_format
        build_dir.mkdir(exist_ok=True, parents=True)
        cmd = [sys.executable, "-m", "sphinx"]

        if output_format == "html":
            cmd.extend(["-b", "html"])
        elif output_format == "pdf":
            cmd.extend(["-b", "latex"])
        elif output_format == "epub":
            cmd.extend(["-b", "epub"])
        else:
            logger.error(f"❌ Unknown output format: {output_format}")
            continue

        cmd.extend([str(DOCS_DIR), str(build_dir)])
        code, _, err = run_command(cmd)

        if code != 0:
            logger.error(f"❌ {output_format.upper()} build failed: {err}")
            return code

        logger.info(f"✅ {output_format.upper()} build completed successfully")

        if output_format == "pdf":
            logger.info("📄 Running LaTeX build to generate PDF")
            code, _, err = run_command(["make", "-C", str(build_dir), "all-pdf"])
            if code != 0:
                logger.error(f"❌ PDF generation failed: {err}")
                return code
            logger.info("✅ PDF generation completed successfully")

    if open_after:
        html_index = BUILD_DIR / "html" / "index.html"
        if html_index.exists():
            logger.info(f"🌐 Opening documentation: {html_index}")
            if sys.platform == "linux":
                run_command(["xdg-open", str(html_index)])
            elif sys.platform == "darwin":
                run_command(["open", str(html_index)])
            elif sys.platform == "win32":
                run_command(["start", str(html_index)], shell=True)

    logger.info(f"📚 Documentation build complete. Output in: {BUILD_DIR}")
    return 0

def cmd_clean(_: argparse.Namespace) -> int:
    logger.info("🧹 Cleaning documentation build artifacts")
    import shutil

    try:
        if BUILD_DIR.exists():
            logger.info(f"🗑️ Removing build directory: {BUILD_DIR}")
            shutil.rmtree(BUILD_DIR, ignore_errors=True)

        doctrees = DOCS_DIR / "_build" / "doctrees"
        if doctrees.exists():
            logger.info(f"🗑️ Removing doctrees: {doctrees}")
            shutil.rmtree(doctrees, ignore_errors=True)

        for pycache in DOCS_DIR.glob("**/__pycache__"):
            if pycache.is_dir():
                logger.debug(f"🗑️ Removing __pycache__: {pycache}")
                shutil.rmtree(pycache, ignore_errors=True)

        logger.info("✅ Clean operation completed successfully")
        return 0
    except Exception as e:
        logger.error(f"❌ Clean operation failed: {e}")
        return 1

def cmd_check(args: argparse.Namespace) -> int:
    logger.info("🔍 Checking documentation for issues")
    markdown_files = list(DOCS_DIR.glob("**/*.md"))
    rst_files = list(DOCS_DIR.glob("**/*.rst"))
    total_files = len(markdown_files) + len(rst_files)
    logger.info(f"📊 Found {total_files} documentation files ({len(markdown_files)} Markdown, {len(rst_files)} RST)")

    logger.info("🔗 Checking for broken references")
    code, _, err = run_command([
        sys.executable, "-m", "sphinx.ext.intersphinx",
        str(DOCS_DIR / "conf.py")
    ])
    if code != 0:
        logger.warning(f"⚠️ Intersphinx check had issues: {err}")

    logger.info("🔍 Running link check")
    code, _, err = run_command([
        sys.executable, "-m", "sphinx", "-b", "linkcheck",
        str(DOCS_DIR), str(BUILD_DIR / "linkcheck")
    ])
    if "broken links found" in err:
        logger.warning("⚠️ Broken links detected")
        for line in err.splitlines():
            if "broken" in line or "error" in line.lower():
                logger.warning(f"  {line}")

    logger.info("⚠️ Running test build with warnings-as-errors")
    code, _, err = run_command([
        sys.executable, "-m", "sphinx", "-b", "html", "-W",
        str(DOCS_DIR), str(BUILD_DIR / "test")
    ])
    if code != 0:
        logger.error("❌ Test build failed - documentation has warnings that would be errors")
        for line in err.splitlines():
            if "WARNING:" in line:
                logger.warning(f"  {line}")
        return code
    else:
        logger.info("✅ Test build passed - documentation has no critical warnings")

    logger.info("✅ Documentation check completed")
    return 0

def cmd_serve(args: argparse.Namespace) -> int:
    port = getattr(args, 'port', 8000)

    code, _, err = run_command([
        sys.executable, "-c",
        "import sphinx_autobuild; print('sphinx-autobuild is available')"
    ])
    if code != 0:
        logger.error("❌ sphinx-autobuild is not available, trying to install it")
        code, _, err = run_command([
            sys.executable, "-m", "pip", "install", "sphinx-autobuild"
        ])
        if code != 0:
            logger.error(f"❌ Failed to install sphinx-autobuild: {err}")
            return code

    logger.info(f"🌐 Starting documentation server on port {port}")
    logger.info("📋 Press Ctrl+C to stop the server")
    cmd = [
        sys.executable, "-m", "sphinx_autobuild",
        str(DOCS_DIR), str(BUILD_DIR / "html"),
        "--port", str(port),
        "--open-browser"
    ]
    try:
        process = subprocess.Popen(cmd)
        process.wait()
        return 0
    except KeyboardInterrupt:
        logger.info("🛑 Documentation server stopped")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to start documentation server: {e}")
        return 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📚 CLI infrastructure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🌀 Doc Forge - Universal Documentation Command System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.add_parser('setup', help='Set up documentation environment')

    build_parser = subparsers.add_parser('build', help='Build documentation')
    build_parser.add_argument('-f', '--formats', nargs='+', choices=['html', 'pdf', 'epub'],
                              help='Output formats to build (default: html)')
    build_parser.add_argument('--fix', action='store_true',
                              help='Fix documentation issues before building')
    build_parser.add_argument('--open', action='store_true',
                              help='Open documentation after building')

    subparsers.add_parser('clean', help='Clean build artifacts')
    subparsers.add_parser('check', help='Check documentation for issues')

    serve_parser = subparsers.add_parser('serve', help='Serve documentation with live reload')
    serve_parser.add_argument('-p', '--port', type=int, default=8000, help='Port to serve on')
    return parser

def main() -> int:
    print(r"""
╔═════════════════════════════════════════════════════════════════════════════════╗
║   ██████╗   █████╗   ██████╗    ███████╗  ██████╗  ██████╗   ██████╗  ███████╗  ║
║   ██╔══██╗ ██╔══██╗ ██╔════╝    ██╔════╝ ██╔═══██╗ ██╔══██╗ ██╔════╝  ██╔════╝  ║
║   ██║  ██║ ██║  ██║ ██║         █████╗   ██║   ██║ ██████╔╝ ██║  ███╗ ██████║   ║
║   ██║  ██║ ██║  ██║ ██║         ██╔══╝   ██║   ██║ ██╔══██╗ ██║   ██║ ██║       ║
║   ██████╔╝ ╚█████╔╝ ╚██████╗    ██║      ╚██████╔╝ ██║  ██║ ╚██████╔╝ ███████╗  ║
║   ╚═════╝   ╚════╝   ╚═════╝    ╚═╝       ╚═════╝  ╚═╝  ╚═╝  ╚═════╝  ╚══════╝  ║
╟─────────────────────────────────────────────────────────────────────────────────╢
║            Eidosian Documentation Command Center                                ║
╚═════════════════════════════════════════════════════════════════════════════════╝
""")

    parser = create_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if args.command == 'setup':
        return cmd_setup(args)
    elif args.command == 'build':
        return cmd_build(args)
    elif args.command == 'clean':
        return cmd_clean(args)
    elif args.command == 'check':
        return cmd_check(args)
    elif args.command == 'serve':
        return cmd_serve(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
