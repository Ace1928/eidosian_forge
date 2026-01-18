#!/usr/bin/env python3
"""Migrate existing documentation to the new structure with Eidosian precision."""
import logging
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Set

# Configure logging for self-awareness
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def collect_documents(
    docs_dir: Path,
    handle_auto: bool,
    recursive: bool
) -> List[Path]:
    """
    Gathers markdown documents from docs_dir (optionally recursively).

    Args:
        docs_dir: Path to the documentation directory
        handle_auto: Whether to include auto-generated docs
        recursive: Whether to scan subdirectories recursively
    Returns:
        List of recognized document paths
    """
    matched_files: List[Path] = []
    pattern = "**/*.md" if recursive else "*.md"
    
    for path_item in docs_dir.glob(pattern):
        if path_item.is_file() and is_recognized_doc(path_item, handle_auto):
            matched_files.append(path_item)
            
    return matched_files

def is_recognized_doc(path_item: Path, handle_auto: bool) -> bool:
    """
    Checks if a file is a recognized markdown document.

    Args:
        path_item: Path to the file
        handle_auto: Whether to include auto-generated docs
    Returns:
        True if the file is a recognized markdown document, False otherwise
    """
    # Only process .md files and explicitly exclude requirements.txt
    if path_item.suffix.lower() != ".md" or path_item.name.lower() == "requirements.txt":
        return False
    
    # Exclude auto-generated docs unless specified
    if not handle_auto and "auto" in path_item.name.lower():
        return False
        
    return True

def ensure_directories(docs_dir: Path, universal_dirs: List[str]) -> None:
    """
    Ensures that the required directories exist.

    Args:
        docs_dir: Path to the documentation directory
        universal_dirs: List of subdirectories to create
    """
    for subdir in universal_dirs:
        full_dir = docs_dir / subdir
        try:
            full_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {full_dir}")
        except Exception as e:
            logger.error(f"Failed to create directory {full_dir}: {e}")

def copy_file(src_path: Path, dest_path: Path, skip_existing: bool) -> bool:
    """
    Copies a file from source to destination.

    Args:
        src_path: Path to the source file
        dest_path: Path to the destination file
        skip_existing: Whether to skip if the destination file exists
    Returns:
        True if the file was copied, False otherwise
    """
    if skip_existing and dest_path.exists():
        logger.info(f"Skipped {src_path.name}: already exists in {dest_path.parent}")
        return False
    if dest_path.exists() and src_path.stat().st_mtime <= dest_path.stat().st_mtime:
        logger.info(f"Skipped {src_path.name}: up-to-date in {dest_path.parent}")
        return False
    try:
        shutil.copy2(src_path, dest_path)
        logger.info(f"Copied {src_path.name} -> {dest_path.parent}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src_path.name} to {dest_path.parent}: {e}")
        return False

def ensure_reference_files(docs_dir: Path, reference_files: Dict[str, str]) -> Set[str]:
    """
    Ensures reference files exist, creating empty ones if needed.

    Args:
        docs_dir: Path to the documentation directory
        reference_files: Mapping of reference files to their target directories
    Returns:
        Set of created file names for tracking
    """
    created_files: Set[str] = set()
    
    for filename, target_dir in reference_files.items():
        source_path = docs_dir / filename
        target_path = docs_dir / target_dir / filename
        
        # Check if the file exists either in source or target location
        if not source_path.exists() and not target_path.exists():
            try:
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create empty file with minimal template content
                with open(target_path, 'w') as f:
                    title = filename.replace('.md', '').replace('_', ' ').title()
                    f.write(f"# {title}\n\n*This is an auto-generated reference document.*\n")
                
                logger.info(f"Created empty reference file: {target_path}")
                created_files.add(filename)
            except Exception as e:
                logger.error(f"Failed to create reference file {target_path}: {e}")
    
    return created_files

def migrate_docs(
    docs_dir: Path = Path("docs"),
    handle_auto: bool = False,
    skip_existing: bool = False,
    config_mapping: Dict[str, str] = {},
    recursive: bool = False
) -> Tuple[int, int, int]:
    """
    Migrate markdown documentation to a new directory structure with Eidosian precision.

    Args:
        docs_dir: Path to the documentation directory
        handle_auto: Whether to move auto-generated docs as well
        skip_existing: Whether to skip if the destination file exists
        config_mapping: Additional content mapping configuration
        recursive: Whether to scan subdirectories recursively
    Returns:
        Tuple of (files_migrated, files_skipped, files_created) for migration statistics
    """
    default_content_mapping = {
        "installation.md":       "source/getting_started/",
        "quickstart.md":         "source/getting_started/",
        "README.md":             "source/getting_started/",
        "examples.md":           "source/examples/",
        "advanced_usage.md":     "source/guides/",
        "conventions.md":        "source/concepts/",
        "contributing.md":       "source/guides/",
        "api_reference.md":      "source/reference/",
    }

    if not config_mapping:
        default_content_mapping.update(config_mapping)

    universal_dirs = [
        "source/getting_started",
        "source/guides",
        "source/concepts",
        "source/reference",
        "source/examples",
        "auto/api",
        "auto/introspected",
        "auto/extracted",
    ]

    if not docs_dir.exists():
        logger.error(f"Documentation directory not found: {docs_dir}")
        return 0, 0, 0

    ensure_directories(docs_dir, universal_dirs)
    
    # Ensure reference files exist before migration starts
    files_created = ensure_reference_files(docs_dir, default_content_mapping)

    files_migrated = 0
    files_skipped = 0

    # Process mapped files
    for md_file, target_dir in default_content_mapping.items():
        src_path = docs_dir / md_file
        if src_path.exists():
            dest_path = docs_dir / target_dir / md_file
            if copy_file(src_path, dest_path, skip_existing):
                files_migrated += 1
            else:
                files_skipped += 1
        else:
            # Skip logging if we just created the file
            if md_file not in files_created:
                logger.info(f"Skipped missing file: {md_file}")
                files_skipped += 1

    # Process unmapped markdown files
    leftover_docs = collect_documents(docs_dir, handle_auto, recursive)

    for doc in leftover_docs:
        if doc.name not in default_content_mapping:
            # Skip processing requirements.txt
            if doc.name.lower() == "requirements.txt":
                logger.info(f"Skipped requirements file: {doc.name}")
                files_skipped += 1
                continue
                
            dest_dir = docs_dir / "source/reference"
            dest_path = dest_dir / doc.name
            if copy_file(doc, dest_path, skip_existing):
                files_migrated += 1
            else:
                files_skipped += 1

    logger.info(f"Migration complete: {files_migrated} files migrated, {files_skipped} files skipped, {len(files_created)} files created")
    return files_migrated, files_skipped, len(files_created)

if __name__ == "__main__":
    """
    Command-Line Usage:
        python doc_migration.py [DOCS_DIR] [OPTIONS]
    """
    parser = argparse.ArgumentParser(description="Eidosian Universal Document Migration Tool")
    parser.add_argument("docs_dir", nargs="?", default="docs", help="Path to docs directory.")
    parser.add_argument("--move-auto", action="store_true", help="Include 'auto' files.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if destination exists.")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively.")
    args = parser.parse_args()

    migrate_docs(
        docs_dir=Path(args.docs_dir),
        handle_auto=args.move_auto,
        skip_existing=args.skip_existing,
        recursive=args.recursive
    )
