#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║               EIDOSIAN DECORATOR APPLICATOR                                    ║
║        Safe, Incremental, Idempotent Decorator Application                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This script safely applies the @eidosian decorator to functions across a codebase.

Features:
- Incremental application (one function at a time)
- Automatic verification (tests after each application)
- Automatic reversion on failure
- Skip already decorated functions
- Configurable failure threshold
- Dry-run mode

Usage:
    # Dry run (show what would be decorated)
    python apply_decorators.py /path/to/code --dry-run
    
    # Apply to single file
    python apply_decorators.py /path/to/code/file.py
    
    # Apply to directory
    python apply_decorators.py /path/to/code --recursive
    
    # With verification command
    python apply_decorators.py /path/to/code --verify "pytest"
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple
import json
import hashlib


@dataclass
class FunctionInfo:
    """Information about a function to decorate."""
    name: str
    file_path: Path
    line_number: int
    end_line: int
    is_method: bool
    is_async: bool
    is_decorated: bool
    existing_decorators: List[str]


@dataclass
class ApplicationResult:
    """Result of decorator application."""
    function: FunctionInfo
    success: bool
    error: Optional[str] = None
    reverted: bool = False


@dataclass
class ApplicationState:
    """State of the application process."""
    total_functions: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    consecutive_failures: int = 0
    results: List[ApplicationResult] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "total": self.total_functions,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": [
                {
                    "function": r.function.name,
                    "file": str(r.function.file_path),
                    "line": r.function.line_number,
                    "success": r.success,
                    "error": r.error,
                    "reverted": r.reverted,
                }
                for r in self.results
            ]
        }


class DecoratorApplicator:
    """
    Safely applies @eidosian decorator to Python functions.
    """
    
    EIDOSIAN_IMPORT = "from eidosian_core import eidosian\n"
    EIDOSIAN_DECORATOR = "@eidosian()"
    
    def __init__(
        self,
        target: Path,
        recursive: bool = False,
        dry_run: bool = False,
        verify_command: Optional[str] = None,
        fail_threshold: int = 3,
        skip_patterns: List[str] = None,
        include_private: bool = False,
        backup_dir: Optional[Path] = None,
    ):
        self.target = Path(target)
        self.recursive = recursive
        self.dry_run = dry_run
        self.verify_command = verify_command
        self.fail_threshold = fail_threshold
        self.skip_patterns = skip_patterns or ["test_", "_test", "conftest"]
        self.include_private = include_private
        self.backup_dir = backup_dir or Path.home() / ".eidosian" / "backups"
        
        self.state = ApplicationState()
        self._backups: dict[Path, Path] = {}
    
    def discover_functions(self) -> List[FunctionInfo]:
        """Discover all functions that could be decorated."""
        functions = []
        
        if self.target.is_file():
            files = [self.target]
        elif self.recursive:
            files = list(self.target.rglob("*.py"))
        else:
            files = list(self.target.glob("*.py"))
        
        for file_path in files:
            # Skip patterns
            if any(p in file_path.name for p in self.skip_patterns):
                continue
            
            try:
                functions.extend(self._parse_file(file_path))
            except SyntaxError as e:
                print(f"  ⚠ Skipping {file_path}: {e}")
        
        return functions
    
    def _parse_file(self, file_path: Path) -> List[FunctionInfo]:
        """Parse a Python file and extract function information."""
        functions = []
        
        source = file_path.read_text()
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private functions unless explicitly included
                if node.name.startswith("_") and not self.include_private:
                    continue
                
                # Check existing decorators
                decorators = [self._get_decorator_name(d) for d in node.decorator_list]
                is_decorated = any("eidosian" in d.lower() for d in decorators)
                
                # Determine if method
                is_method = self._is_method(node, tree)
                
                functions.append(FunctionInfo(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    is_method=is_method,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    is_decorated=is_decorated,
                    existing_decorators=decorators,
                ))
        
        return functions
    
    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_decorator_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is a method (inside a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of a file."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
        backup_name = f"{file_path.stem}_{timestamp}_{content_hash}.py.bak"
        
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        
        self._backups[file_path] = backup_path
        return backup_path
    
    def restore_file(self, file_path: Path) -> bool:
        """Restore file from backup."""
        backup_path = self._backups.get(file_path)
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, file_path)
            return True
        return False
    
    def apply_decorator(self, func: FunctionInfo) -> bool:
        """Apply @eidosian decorator to a single function."""
        file_path = func.file_path
        lines = file_path.read_text().splitlines(keepends=True)
        
        # Find the function line (accounting for decorators)
        func_start = func.line_number - 1
        
        # Find where to insert (before first decorator or def line)
        insert_line = func_start
        while insert_line > 0:
            prev_line = lines[insert_line - 1].strip()
            if prev_line.startswith("@"):
                insert_line -= 1
            else:
                break
        
        # Get indentation
        def_line = lines[func_start]
        indent = len(def_line) - len(def_line.lstrip())
        
        # Insert decorator
        decorator_line = " " * indent + self.EIDOSIAN_DECORATOR + "\n"
        lines.insert(insert_line, decorator_line)
        
        # Ensure import exists
        lines = self._ensure_import(lines)
        
        # Write back
        file_path.write_text("".join(lines))
        return True
    
    def _ensure_import(self, lines: List[str]) -> List[str]:
        """Ensure eidosian_core import exists."""
        # Check if import already exists
        for line in lines:
            if "from eidosian_core import" in line and "eidosian" in line:
                return lines
        
        # Find best place to insert import
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_pos = i + 1
            elif stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
                break
        
        lines.insert(insert_pos, self.EIDOSIAN_IMPORT)
        return lines
    
    def verify(self) -> bool:
        """Run verification command."""
        if not self.verify_command:
            return True
        
        try:
            result = subprocess.run(
                self.verify_command,
                shell=True,
                capture_output=True,
                timeout=60,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def run(self) -> ApplicationState:
        """Run the decorator application process."""
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║           EIDOSIAN DECORATOR APPLICATOR                       ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        
        # Discover functions
        print(f"📂 Scanning: {self.target}")
        functions = self.discover_functions()
        
        # Filter already decorated
        to_decorate = [f for f in functions if not f.is_decorated]
        
        self.state.total_functions = len(functions)
        self.state.skipped = len(functions) - len(to_decorate)
        
        print(f"   Found: {len(functions)} functions")
        print(f"   Already decorated: {self.state.skipped}")
        print(f"   To decorate: {len(to_decorate)}")
        print()
        
        if self.dry_run:
            print("🔍 DRY RUN - No changes will be made\n")
            for func in to_decorate:
                print(f"   Would decorate: {func.file_path}:{func.line_number} {func.name}()")
            return self.state
        
        # Process each function
        files_modified: Set[Path] = set()
        
        for func in to_decorate:
            # Check failure threshold
            if self.state.consecutive_failures >= self.fail_threshold:
                print(f"\n⛔ Stopping: {self.fail_threshold} consecutive failures")
                break
            
            print(f"📝 {func.file_path.name}:{func.line_number} {func.name}()...", end=" ")
            
            # Backup file if not already backed up
            if func.file_path not in files_modified:
                self.backup_file(func.file_path)
                files_modified.add(func.file_path)
            
            try:
                # Apply decorator
                self.apply_decorator(func)
                
                # Verify
                if self.verify():
                    print("✓")
                    self.state.succeeded += 1
                    self.state.consecutive_failures = 0
                    self.state.results.append(ApplicationResult(func, True))
                else:
                    # Verification failed - revert
                    self.restore_file(func.file_path)
                    print("✗ (reverted)")
                    self.state.failed += 1
                    self.state.consecutive_failures += 1
                    self.state.results.append(ApplicationResult(
                        func, False, "Verification failed", reverted=True
                    ))
                    
            except Exception as e:
                # Error - revert
                self.restore_file(func.file_path)
                print(f"✗ {e}")
                self.state.failed += 1
                self.state.consecutive_failures += 1
                self.state.results.append(ApplicationResult(
                    func, False, str(e), reverted=True
                ))
            
            self.state.processed += 1
        
        # Summary
        print()
        print("═══════════════════════════════════════════════════════════════")
        print(f"  Processed: {self.state.processed}")
        print(f"  Succeeded: {self.state.succeeded}")
        print(f"  Failed:    {self.state.failed}")
        print(f"  Skipped:   {self.state.skipped}")
        print("═══════════════════════════════════════════════════════════════")
        
        return self.state


def main():
    parser = argparse.ArgumentParser(
        description="Apply @eidosian decorator to Python functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "target",
        type=Path,
        help="File or directory to process",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Command to run after each modification to verify correctness",
    )
    parser.add_argument(
        "--fail-threshold",
        type=int,
        default=3,
        help="Stop after N consecutive failures (default: 3)",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private functions (starting with _)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        help="Patterns to skip (default: test_, _test, conftest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    
    args = parser.parse_args()
    
    applicator = DecoratorApplicator(
        target=args.target,
        recursive=args.recursive,
        dry_run=args.dry_run,
        verify_command=args.verify,
        fail_threshold=args.fail_threshold,
        include_private=args.include_private,
        skip_patterns=args.skip if args.skip else None,
    )
    
    state = applicator.run()
    
    if args.output:
        args.output.write_text(json.dumps(state.to_dict(), indent=2))
        print(f"\nResults saved to: {args.output}")
    
    sys.exit(0 if state.failed == 0 else 1)


if __name__ == "__main__":
    main()
