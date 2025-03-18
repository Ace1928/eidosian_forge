"""
Version update operations with file system integration.
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

from ..core.config import VersionConfig

logger = logging.getLogger("forge.version")

class CompleteVersionUpdateResult(TypedDict):
    """Extended version update result with version reference information."""
    updated: bool
    files_changed: List[str]
    files_examined: int
    duration_seconds: float
    previous_version: str
    current_version: str

def update_version(new_version: str,
                   current_version: str,
                   repo_path: Optional[Path] = None,
                   config: Optional[VersionConfig] = None) -> CompleteVersionUpdateResult:
    """Update version references throughout codebase with surgical precision."""
    start_time: datetime = datetime.now()
    repo_root: Path = repo_path or Path.cwd()

    # Fast path for no-op updates - avoid unnecessary file operations
    if current_version == new_version:
        return {
            "updated": False,
            "files_changed": [],
            "files_examined": 0,
            "duration_seconds": 0.0,
            "previous_version": current_version,
            "current_version": new_version
        }

    # Validate version format with fail-fast principle
    if not (version_match := re.match(r'^(\d+)\.(\d+)\.(\d+)', new_version)):
        raise ValueError(f"Invalid version format: {new_version}")

    # Parse components once for efficiency
    new_major, new_minor, new_patch = map(int, version_match.groups())

    # Get current components if config is provided
    current_major, current_minor, current_patch = 0, 0, 0
    if config:
        current_major, current_minor, current_patch = config.major, config.minor, config.patch

    files_updated: List[str] = []
    files_examined: int = 0

    # Surgical replacement patterns with capture groups for precise changes
    version_patterns: List[Tuple[str, str]] = [
        (rf'(version\s*=\s*["\']){re.escape(current_version)}(["\'])',
         rf'\g<1>{new_version}\g<2>'),
        (rf'(__version__\s*=\s*["\']){re.escape(current_version)}(["\'])',
         rf'\g<1>{new_version}\g<2>'),
        (rf'(VERSION_MAJOR\s*=\s*){current_major}',
         rf'\g<1>{new_major}'),
        (rf'(VERSION_MINOR\s*=\s*){current_minor}',
         rf'\g<1>{new_minor}'),
        (rf'(VERSION_PATCH\s*=\s*){current_patch}',
         rf'\g<1>{new_patch}'),
    ]

    # Exclusion rules for efficiency and safety
    skip_dirs: set[str] = {
        "__pycache__", "dist", "build", "venv", ".venv",
        ".git", "node_modules"
    }
    valid_extensions: set[str] = {
        ".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".cfg"
    }

    def update_file(path: Path) -> bool:
        """Update version references in a file, return True if changed"""
        nonlocal files_examined
        files_examined += 1

        try:
            content: str = path.read_text(encoding="utf-8")
            new_content: str = content

            for pattern, replacement in version_patterns:
                new_content = re.sub(pattern, replacement, new_content)

            # Idempotent write - only modify when necessary
            if new_content != content:
                path.write_text(new_content, encoding="utf-8")
                files_updated.append(str(path.relative_to(repo_root)))
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed updating {path}: {e}")
            return False

    # Process files with efficient filtering
    for file_path in repo_root.rglob('*'):
        # Fast-path exclusions first to minimize work
        if (file_path.is_dir() or
            file_path.name.startswith('.') or
            any(p in file_path.parts for p in skip_dirs)):
            continue

        # Process only relevant file types
        if file_path.suffix.lower() in valid_extensions:
            update_file(file_path)

    # Special handling for critical configuration files
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        update_file(pyproject_path)

    # Calculate operation duration for metrics
    duration: float = (datetime.now() - start_time).total_seconds()

    # Update runtime state for consistency
    if files_updated and config:
        config.update(
            __version__=new_version,
            major=new_major, minor=new_minor, patch=new_patch
        )

        # Message with contextually appropriate wit
        log_message = f"Updated {len(files_updated)} files to {new_version}"
        if duration < 1.0:
            logger.info(f"{log_message} faster than a quantum fluctuation ⚡")
        elif duration > 5.0:
            logger.info(f"{log_message} with the inevitability of entropy ♾️")
        else:
            logger.info(f"{log_message} ✓")

        # Detailed logging only when appropriate
        if logger.level <= logging.DEBUG:
            shown = files_updated[:3]
            if len(files_updated) > 3:
                shown.append(f"...and {len(files_updated) - 3} more")
            logger.debug("Modified: " + ", ".join(shown))
    else:
        logger.info("No version references found - perfect stealth or nothing to change")

    return {
        "updated": bool(files_updated),
        "files_changed": files_updated,
        "files_examined": files_examined,
        "duration_seconds": duration,
        "previous_version": current_version,
        "current_version": new_version
    }

# API stability through semantic aliasing
update_version_universally = update_version
