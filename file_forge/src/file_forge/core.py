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
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from .library import FileLibraryDB, index_path

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
        rg_binary = shutil.which("rg")
        if rg_binary:
            try:
                proc = subprocess.run(
                    [rg_binary, "-l", "--fixed-strings", "--no-messages", pattern, str(search_dir)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # rg: 0=match found, 1=no matches, >1 error
                if proc.returncode in (0, 1):
                    paths = [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
                    return sorted(paths)
            except Exception:
                # Fall back to Python scan when rg execution fails.
                pass

        matches: List[Path] = []
        for root, _, files in os.walk(search_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    if pattern in file_path.read_text(errors='ignore'):
                        matches.append(file_path)
                except Exception:
                    pass
        return sorted(matches)

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

    def default_library_path(self) -> Path:
        return (self.base_path / ".file_forge" / "library.sqlite").resolve()

    @eidosian()
    def index_directory(
        self,
        directory: Path,
        *,
        db_path: Optional[Path] = None,
        remove_after_ingest: bool = False,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        target_dir = Path(directory).resolve()
        db = FileLibraryDB(db_path or self.default_library_path())
        indexed = 0
        skipped = 0
        removed = 0
        files: List[Dict[str, Any]] = []
        for file_path in sorted(target_dir.rglob("*")):
            if not file_path.is_file():
                continue
            result = index_path(db=db, file_path=file_path)
            files.append(result)
            if result.get("status") == "indexed":
                indexed += 1
                if remove_after_ingest:
                    file_path.unlink()
                    removed += 1
            else:
                skipped += 1
            if max_files is not None and len(files) >= max(0, int(max_files)):
                break
        return {
            "status": "success",
            "directory": str(target_dir),
            "db_path": str(Path(db_path or self.default_library_path()).resolve()),
            "indexed": indexed,
            "skipped": skipped,
            "removed": removed,
            "results": files,
        }

    @eidosian()
    def restore_indexed_file(
        self,
        file_path: Path,
        *,
        target_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> Path:
        db = FileLibraryDB(db_path or self.default_library_path())
        source = Path(file_path).resolve()
        target = Path(target_path).resolve() if target_path is not None else source
        return db.restore_file(file_path=source, target_path=target)


    @eidosian()
    def restore_directory(
        self,
        source_root: Path,
        *,
        target_root: Path,
        db_path: Optional[Path] = None,
        overwrite: bool = False,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        db = FileLibraryDB(db_path or self.default_library_path())
        source_root = Path(source_root).resolve()
        target_root = Path(target_root).resolve()
        restored = 0
        skipped_existing = 0
        overwritten_existing = 0
        missing_records = 0
        missing_blobs = 0
        results: List[Dict[str, Any]] = []
        count = 0

        for record in db.iter_file_records(path_prefix=source_root):
            if max_files is not None and count >= max(0, int(max_files)):
                break
            count += 1
            path_text = str(record.get("file_path") or "")
            if not path_text:
                continue
            original = Path(path_text).resolve()
            try:
                rel_path = original.relative_to(source_root)
            except ValueError:
                continue
            target = (target_root / rel_path).resolve()
            if target.exists():
                if not overwrite:
                    skipped_existing += 1
                    results.append({
                        "status": "skipped_existing",
                        "file_path": str(original),
                        "target_path": str(target),
                    })
                    continue
                overwritten_existing += 1
            if not record.get("content_hash"):
                missing_records += 1
                results.append({
                    "status": "missing_record",
                    "file_path": str(original),
                    "target_path": str(target),
                })
                continue
            blob = db.get_blob(str(record.get("content_hash") or ""))
            if blob is None:
                missing_blobs += 1
                results.append({
                    "status": "missing_blob",
                    "file_path": str(original),
                    "target_path": str(target),
                    "content_hash": str(record.get("content_hash") or ""),
                })
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            restored_path = db.restore_file(file_path=original, target_path=target)
            restored += 1
            results.append({
                "status": "restored",
                "file_path": str(original),
                "target_path": str(restored_path),
            })

        return {
            "status": "success",
            "source_root": str(source_root),
            "target_root": str(target_root),
            "db_path": str(Path(db_path or self.default_library_path()).resolve()),
            "restored": restored,
            "skipped_existing": skipped_existing,
            "overwritten_existing": overwritten_existing,
            "missing_records": missing_records,
            "missing_blobs": missing_blobs,
            "results": results,
        }

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
