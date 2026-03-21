from eidosian_core import eidosian
"""
File Forge CLI - File operations and management.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List

from file_forge import FileForge

@eidosian()
def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="file-forge",
        description="📁 File Forge - File operations and management",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Info
    info_parser = subparsers.add_parser("info", help="Get file information")
    info_parser.add_argument("path", help="File path")
    
    # Hash
    hash_parser = subparsers.add_parser("hash", help="Compute file hash")
    hash_parser.add_argument("path", help="File path")
    hash_parser.add_argument("--algorithm", default="sha256", help="Hash algorithm")
    
    # Tree
    tree_parser = subparsers.add_parser("tree", help="Show directory tree")
    tree_parser.add_argument("path", nargs="?", default=".", help="Directory path")
    tree_parser.add_argument("--depth", type=int, default=3, help="Max depth")
    
    # Status
    subparsers.add_parser("status", help="Show file forge status")

    index_parser = subparsers.add_parser("index", help="Index files into the reversible file library")
    index_parser.add_argument("path", nargs="?", default=".", help="Directory path")
    index_parser.add_argument("--db-path", help="SQLite file library path")
    index_parser.add_argument("--remove-after-ingest", action="store_true", help="Capture then remove files")
    index_parser.add_argument("--max-files", type=int, help="Maximum files to process")

    restore_parser = subparsers.add_parser("restore", help="Restore a file from the file library")
    restore_parser.add_argument("file_path", help="Original indexed file path")
    restore_parser.add_argument("--target", help="Restore destination path")
    restore_parser.add_argument("--db-path", help="SQLite file library path")


    restore_tree_parser = subparsers.add_parser("restore-tree", help="Restore a directory subtree from the file library")
    restore_tree_parser.add_argument("source_root", help="Original indexed source root")
    restore_tree_parser.add_argument("target_root", help="Destination root")
    restore_tree_parser.add_argument("--db-path", help="SQLite file library path")
    restore_tree_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    restore_tree_parser.add_argument("--max-files", type=int, help="Maximum files to restore")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("File Forge v0.1.0")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    forge = FileForge()

    if args.command == "status":
        print("📁 File Forge Status")
        print("  Status: operational")
        print(f"  Library: {forge.default_library_path()}")
        return 0
    
    elif args.command == "info":
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist")
            return 1
        stat = path.stat()
        print(f"📄 {path}")
        print(f"  Size: {stat.st_size:,} bytes")
        print(f"  Modified: {stat.st_mtime}")
        print(f"  Type: {'directory' if path.is_dir() else 'file'}")
        return 0
    
    elif args.command == "hash":
        import hashlib
        path = Path(args.path)
        if not path.exists():
            print(f"Error: {path} does not exist")
            return 1
        h = hashlib.new(args.algorithm)
        h.update(path.read_bytes())
        print(f"{args.algorithm}: {h.hexdigest()}")
        return 0
    
    elif args.command == "index":
        result = forge.index_directory(
            Path(args.path),
            db_path=Path(args.db_path) if args.db_path else None,
            remove_after_ingest=bool(args.remove_after_ingest),
            max_files=args.max_files,
        )
        print(f"Indexed: {result['indexed']}  Skipped: {result['skipped']}  Removed: {result['removed']}")
        return 0

    elif args.command == "restore":
        restored = forge.restore_indexed_file(
            Path(args.file_path),
            target_path=Path(args.target) if args.target else None,
            db_path=Path(args.db_path) if args.db_path else None,
        )
        print(restored)
        return 0

    elif args.command == "restore-tree":
        result = forge.restore_directory(
            Path(args.source_root),
            target_root=Path(args.target_root),
            db_path=Path(args.db_path) if args.db_path else None,
            overwrite=bool(args.overwrite),
            max_files=args.max_files,
        )
        print(
            "Restored: "
            f"{result['restored']}  Skipped existing: {result['skipped_existing']}  "
            f"Missing records: {result['missing_records']}  Missing blobs: {result['missing_blobs']}"
        )
        return 0

    elif args.command == "tree":
        @eidosian()
        def print_tree(p: Path, prefix: str = "", depth: int = 0, max_depth: int = 3):
            if depth > max_depth:
                return
            print(f"{prefix}{p.name}/" if p.is_dir() else f"{prefix}{p.name}")
            if p.is_dir():
                children = sorted(p.iterdir())
                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    connector = "└── " if is_last else "├── "
                    print(f"{prefix}{connector}{child.name}{'/' if child.is_dir() else ''}")
                    if child.is_dir() and depth < max_depth:
                        print_tree(child, new_prefix, depth + 1, max_depth)
        
        path = Path(args.path)
        print_tree(path, max_depth=args.depth)
        return 0
    
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
