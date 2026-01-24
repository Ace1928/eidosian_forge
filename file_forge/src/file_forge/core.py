from eidosian_core import eidosian
"""
File Forge - Pattern-aware filesystem intelligence.
Provides organization optimization and content-based categorization.
"""
import os
import shutil
import logging
import hashlib
import fnmatch
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

class FileForge:
    """
    Manages filesystem operations with Eidosian precision.
    Includes hashing, deduplication, and synchronization.
    """
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()

    @eidosian()
    def calculate_hash(self, file_path: Path) -> str:
        """Calculate the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @eidosian()
    def find_duplicates(self, directory: Path) -> Dict[str, List[Path]]:
        """Find duplicate files in a directory based on their content hash."""
        hashes: Dict[str, List[Path]] = {}
        for root, _, files in os.walk(directory):
            for file in files:
                path = Path(root) / file
                file_hash = self.calculate_hash(path)
                if file_hash not in hashes:
                    hashes[file_hash] = []
                hashes[file_hash].append(path)
        
        return {h: paths for h, paths in hashes.items() if len(paths) > 1}

    @eidosian()
    def sync_directories(self, source: Path, target: Path):
        """Mirror source directory to target (one-way sync)."""
        if not target.exists():
            target.mkdir(parents=True)
            
        for item in source.iterdir():
            target_item = target / item.name
            if item.is_dir():
                self.sync_directories(item, target_item)
            else:
                if not target_item.exists() or self.calculate_hash(item) != self.calculate_hash(target_item):
                    shutil.copy2(item, target_item)

    @eidosian()
    def categorize_files(self, target_dir: Path) -> Dict[str, List[str]]:
        """Categorize files in a directory by their extension."""
        categories = {}
        for item in target_dir.iterdir():
            if item.is_file():
                ext = item.suffix.lstrip('.').lower() or "no_extension"
                if ext not in categories:
                    categories[ext] = []
                categories[ext].append(item.name)
        return categories

    @eidosian()
    def search_content(self, pattern: str, directory: Optional[Path] = None) -> List[Path]:
        """Search for files containing a specific string pattern."""
        search_dir = directory or self.base_path
        matches = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    if pattern in file_path.read_text(errors='ignore'):
                        matches.append(file_path)
                except Exception:
                    pass
        return matches

    @eidosian()
    def find_files(self, pattern: str, directory: Optional[Path] = None) -> List[Path]:
        """Search for files by filename pattern (glob)."""
        search_dir = directory or self.base_path
        matches = []
        for root, _, files in os.walk(search_dir):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(Path(root) / name)
        return matches

    @eidosian()
    def ensure_structure(self, structure: Dict[str, Any], root: Optional[Path] = None):
        """Recursively ensure a directory structure exists."""
        current_root = root or self.base_path
        for name, content in structure.items():
            path = current_root / name
            if isinstance(content, dict):
                path.mkdir(exist_ok=True, parents=True)
                self.ensure_structure(content, path)
            else:
                path.touch(exist_ok=True)
                if content:
                    path.write_text(str(content))
        return True