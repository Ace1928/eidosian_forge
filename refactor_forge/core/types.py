"""
Type Definitions - Structural Typing System 📊

Provides precise type definitions for all data structures used throughout
the Eidosian Refactor system, ensuring type safety and clarity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Any, TypedDict

# ╭──────────────────────────────────────────────────────╮
# │  🏗️ Core Data Structures - Foundational Types       │
# ╰──────────────────────────────────────────────────────╯

@dataclass
class RefactorOptions:
    """Options for controlling the refactoring process."""
    source_path: Union[str, Path]
    output_dir: Optional[Union[str, Path]] = None
    package_name: Optional[str] = None
    analyze_only: bool = False
    dry_run: bool = False
    verbose: bool = False


class FileInfo(TypedDict):
    """Information about a source file."""
    path: str
    size: int
    name: str
    stem: str


class FunctionInfo(TypedDict):
    """Information about a function."""
    name: str
    lineno: int
    args: List[str]
    docstring: str


class ClassInfo(TypedDict):
    """Information about a class."""
    name: str
    lineno: int
    methods: List[str]
    docstring: str


class ModuleInfo(TypedDict, total=False):
    """Information about an identified module."""
    name: str
    start_line: int
    end_line: int
    content: str
    purpose: str
    docstring: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]


class AnalysisResult(TypedDict):
    """Results of code analysis."""
    modules: List[ModuleInfo]
    dependencies: Any  # NetworkX graph
    symbols: Dict[str, Dict[str, Any]]
    file_info: FileInfo


class TransformationFile(TypedDict):
    """Information about a generated file."""
    path: str
    content: str


class TransformationResult(TypedDict):
    """Results of code transformation."""
    output_path: str
    files: List[TransformationFile]
    package_name: str
    module_map: Dict[str, str]
