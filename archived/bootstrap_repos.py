#!/usr/bin/env python3
"""EIDOSIAN FORGE REPOSITORY BOOTSTRAPPER v4.0

Absolute zero‑friction repository instantiation—engineered for perfect execution.
Equipped with militarized credential security protocols for maximum protection.
"""

import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


# ⚡️ ANSI COLOR ENGINE - OPTIMIZED FOR COGNITIVE IMPACT
def print_color(text: str, color_code: str) -> None:
    """Project information with chromatic precision for maximum cognitive impact."""
    print(f"\033[{color_code}m{text}\033[0m")


# 🔍 DIAGNOSTIC MATRIX - HYPEROPTIMIZED OBSERVABILITY
def enable_eidosian_diagnostics() -> Path:
    """Activate comprehensive diagnostic protocols and return active log path."""
    debug_dir = Path.home() / "repos" / "eidosian_diagnostics"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ.update({
        "PYTHONVERBOSE": "2",
        "EIDOSIAN_DEBUG": "1"
    })
    
    log_file = debug_dir / f"forge_bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.environ["EIDOSIAN_DIAGNOSTIC_LOG"] = str(log_file)
    
    print_color("🔍 Eidosian diagnostics enabled", "38;5;51")
    print_color(f"   Log: {log_file}", "38;5;245")
    
    return log_file


# ⚔️ EXCEPTION NEUTRALIZER - STRUCTURED ERROR CONTROL
def eidosian_exception_handler(exc_type, exc_value, exc_traceback):
    """Crush exceptions with extreme prejudice, leaving perfect diagnostic footprints."""
    print_color(f"ERROR: {exc_type.__name__}: {exc_value}", "1;31")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    log_path = os.environ.get("EIDOSIAN_DIAGNOSTIC_LOG")
    if log_path:
        try:
            with open(log_path, "a") as log:
                log.write(f"\n⚠️ EXCEPTION: {exc_type.__name__}: {exc_value}\n")
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=log)
        except Exception:
            # Failed silently - can't interrupt critical path for logging
            pass
    sys.exit(1)


# 🛡️ GIT REPOSITORY FORTRESS - ISOLATION PROTOCOL
def ensure_git_isolation(repo_path: Path) -> None:
    """Establish impenetrable boundaries between repositories.
    
    Args:
        repo_path: Target repository requiring isolation
    """
    # Secure local .gitignore
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = f.read()
            
        # Inject isolation directives if not present
        if "../*_forge_repo/" not in content and "../*_repo/" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n# Exclude sibling repositories\n../*/\n!../README.md\n")
                print_color(f"✓ Repository isolation enforced: {repo_path.name}", "38;5;82")
    
    # Initialize git repository if absent
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=repo_path, stdout=subprocess.PIPE)
        print_color(f"✓ Git repository initialized: {repo_path.name}", "38;5;82")
    
    # Enforce consistent permission handling
    subprocess.run(
        ["git", "config", "core.filemode", "false"],
        cwd=repo_path, 
        stdout=subprocess.PIPE
    )


# 🔐 CREDENTIAL SECURITY MATRIX - RUTHLESS PROTECTION
def enforce_credential_security(repo_path: Path) -> None:
    """Deploy impenetrable credential security protocols with extreme prejudice.
    
    Args:
        repo_path: Repository requiring credential security
    """
    # 1. SECURITY PATTERNS - THE EXCLUSION MATRIX
    credential_patterns = [
        # Core credential files
        ".pypirc", "**/.pypirc", 
        # Encryption keys
        "**/*.key", "**/*.pem", "**/*.token",
        # Environment files
        "**/.env", "**/.env.*",
        # Generic credential stores
        "**/credentials.*", "**/secrets.*"
    ]
    
    # 2. GITIGNORE FORTIFICATION
    gitignore_path = repo_path / ".gitignore"
    
    # Read existing or initialize new
    content = []
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = [line.strip() for line in f.read().splitlines()]
    
    # Inject missing security patterns - no duplicates
    modified = False
    for pattern in credential_patterns:
        if pattern not in content:
            content.append(pattern)
            modified = True
    
    # Write only if changes made - avoid unnecessary I/O
    if modified:
        with open(gitignore_path, "w") as f:
            f.write("\n".join(content) + "\n")
        print_color(f"🔒 Credential exclusion matrix deployed: {repo_path.name}", "38;5;82")
    
    # 3. GLOBAL PROTECTION BARRIER
    global_excludes = Path.home() / ".git" / "info" / "excludes"
    global_excludes.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify global exclusions
    global_content = ""
    if global_excludes.exists():
        global_content = global_excludes.read_text()
    
    # Add global patterns only if missing
    if ".pypirc" not in global_content:
        with open(global_excludes, "a") as f:
            f.write("\n# EIDOSIAN CREDENTIAL VAULT\n.pypirc\n**/.pypirc\n*.key\n*.token\n.env\n")
        print_color("🔒 Global credential protection activated", "38;5;82")
    
    # 4. ACTIVE DECONTAMINATION - REMOVE TRACKED CREDENTIALS
    pypirc_path = repo_path / ".pypirc"
    if pypirc_path.exists():
        try:
            # Untrack without deletion - preserve local config
            subprocess.run(
                ["git", "rm", "--cached", ".pypirc"],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print_color(f"⚠️ Credential file quarantined: {repo_path.name}", "38;5;208")
        except subprocess.SubprocessError:
            # Already untracked or not in git - safe state
            pass
    
    # 5. SECURE TEMPLATE DEPLOYMENT
    templates_dir = repo_path / "templates" / "credentials"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Create credential templates with security warnings
    pypirc_example = templates_dir / "pypirc.example"
    if not pypirc_example.exists():
        with open(pypirc_example, "w") as f:
            f.write("""# EIDOSIAN SECURITY PROTOCOL
# ⚠️  TEMPLATE ONLY - NEVER COMMIT REAL CREDENTIALS ⚠️
# Copy to ~/.pypirc (HOME directory) and add real tokens there

[pypi]
username = __token__
password = pypi-PLACEHOLDER_TOKEN_REPLACE_ME

[testpypi]
username = __token__
password = pypi-PLACEHOLDER_TOKEN_REPLACE_ME
""")
        print_color(f"📄 Secure credential template deployed: {repo_path.name}", "38;5;117")
    
    # Security documentation
    readme_path = templates_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write("""# ⚠️ EIDOSIAN CREDENTIAL SECURITY PROTOCOL ⚠️

## ZERO-TOLERANCE SECURITY POLICY

Templates in this directory contain **PLACEHOLDER VALUES ONLY**.
Real credentials must **NEVER** appear in version control.

## CREDENTIAL DEPLOYMENT PROCEDURE:

1. Copy templates to secure locations **OUTSIDE** this repository
   - PyPI credentials: `~/.pypirc`
   - Environment variables: `~/.env` or through your CI system
   - API keys: Secure credential store only

2. Replace placeholders with real values in the copied files

3. **NEVER** commit files containing real credentials

## AUTOMATIC SECURITY MEASURES:

- Repository-level credential exclusion (.gitignore)
- Global Git exclusion patterns
- Automatic untracking of detected credential files

If you discover committed credentials, revoke them **IMMEDIATELY**
and rewrite Git history to remove all traces.
""")


def main() -> None:
    """Execute comprehensive repository bootstrapping with credential security."""
    # Initialize diagnostic matrix
    log_path = enable_eidosian_diagnostics()
    sys.excepthook = eidosian_exception_handler

    # Display execution header
    border = "════════════════════════════════════════════"
    print_color(border, "38;5;51")
    print_color("🔥 EIDOSIAN FORGE REPOSITORY BOOTSTRAPPER 🔥", "38;5;196")
    print_color(border, "38;5;51")

    # Navigate to operational directory
    repos_dir = Path.home() / "repos"
    try:
        os.chdir(repos_dir)
    except Exception:
        print_color("⚠️ FATAL: Directory navigation failure", "1;31")
        sys.exit(1)

    # Repository configuration matrix
    REPOS: List[Tuple[str, str]] = [
        ("doc_forge_repo", "Documentation generation tools"),
        ("file_forge_repo", "File system management utilities"),
        ("knowledge_forge_repo", "Knowledge base management system"),
        ("memory_forge_repo", "Memory and state management tools"),
        ("metadata_forge_repo", "Metadata extraction and management"),
        ("ollama_forge_repo", "Ollama integration utilities"),
        ("refactor_forge_repo", "Code refactoring automation"),
        ("terminal_forge_repo", "Terminal UI and management tools"),
    ]

    success_count = 0
    failed: List[str] = []

    # Validate operational dependencies
    try:
        import repo_forge  # noqa: F401
    except ImportError:
        print_color("❌ DEPENDENCY ERROR: repo_forge not in Python path", "1;31")
        print_color("💡 SOLUTION: pip install -e ~/repos/repo_forge_repo", "38;5;220")
        sys.exit(1)

    # Import operational components
    try:
        from repo_forge.cli import main as forge_main
    except ImportError:
        print_color("❌ ERROR: Unable to import repo_forge.cli.main", "1;31")
        sys.exit(1)

    # Process target repositories
    for repo_name, description in REPOS:
        print()  # Operational clarity
        print_color(f"🚀 TARGET: {repo_name}", "1;36")
        print_color(f"📝 DESCRIPTOR: {description}", "38;5;245")

        # Log operational context
        if log_path:
            try:
                with open(log_path, "a") as log:
                    log.write(f"\n\n🚀 Bootstrap execution for {repo_name} at {datetime.now().isoformat()}\n")
            except Exception:
                pass  # Non-critical path - continue without logging if necessary

        repo_path = repos_dir / repo_name
        
        try:
            # Prepare operational parameters
            args = ["--name", repo_name, "--output", ".", "--safe-mode", "--languages", "python"]
            
            # Execute bootstrap operation
            result = forge_main(args)
            
            # Process operation outcome
            if result is None or result == 0:
                print_color(f"✅ BOOTSTRAPPED: {repo_name}", "1;32")
                
                # Deploy security matrix
                ensure_git_isolation(repo_path)
                enforce_credential_security(repo_path)
                
                success_count += 1
            else:
                print_color(f"❌ OPERATION FAILED: {repo_name}", "1;31")
                failed.append(repo_name)
        except Exception:
            print_color(f"❌ OPERATION FAILED: {repo_name}", "1;31")
            traceback.print_exc()
            failed.append(repo_name)

    # Operational summary
    print()
    print_color(border, "38;5;51")
    print_color("📊 OPERATION SUMMARY", "1;34")
    print_color(border, "38;5;51")
    print(f"✅ SUCCESSFUL OPERATIONS: \033[1;32m{success_count}\033[0m/\033[1;34m{len(REPOS)}\033[0m")
    
    # Status reporting
    if failed:
        print_color("❌ FAILED OPERATIONS:", "1;31")
        for repo in failed:
            print_color(f"   → {repo}", "38;5;196")
        sys.exit(1)
    else:
        print_color("🏆 ALL TARGETS SUCCESSFULLY BOOTSTRAPPED", "38;5;82")
        print_color(f"📁 Development environment ready at: {os.getcwd()}", "38;5;87")
        print_color("🔒 CREDENTIAL SECURITY PROTOCOLS ACTIVATED", "38;5;226")


if __name__ == "__main__":
    main()