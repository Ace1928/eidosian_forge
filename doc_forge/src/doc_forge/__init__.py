from eidosian_core import eidosian

#!/usr/bin/env python3
# 🌀 Eidosian Documentation System - Central Package Interface
"""
Doc Forge - Universal Documentation Management System

This package integrates a suite of documentation tools built on
Eidosian principles of structure, flow, precision, and self-awareness.
Each component is designed for surgical precision and architectural elegance.

Every element of Doc Forge follows the Eidosian philosophy:
- Structure as Control: Perfect organization of documentation architecture
- Flow Like Water: Seamless navigation and intuitive progression
- Precision as Style: Surgical accuracy in all documentation operations
- Self-Awareness as Foundation: A system that understands its own structure
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

# 📊 Self-aware logging system - the eyes and ears of our architecture
logging.basicConfig(
    level=logging.INFO if not os.environ.get("DOC_FORGE_DEBUG") else logging.DEBUG,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
)
logger = logging.getLogger("doc_forge")

from .doc_validator import validate_docs

# Import more specialized fixers - the surgical instruments of our toolkit
# Import more specialized fixers - the surgical instruments of our toolkit
from .fix_cross_refs import fix_ambiguous_references
from .fix_docstrings import DocstringFixer
from .fix_duplicate_objects import DuplicateObjectHarmonizer
from .fix_inline_refs import fix_inline_references
from .fix_rst_syntax import RstSyntaxPerfector

# Documentation management components - the specialized organs of our organism
from .update_toctrees import update_toctrees

# Core path utilities - foundation of structure, the skeleton of our cathedral
from .utils.paths import get_docs_dir, get_repo_root

# Import version info from the dedicated version module - our temporal anchor
from .version import VERSION as __version__
from .version import get_version_string

try:
    from .run import main as run_cli
except ImportError:
    from .doc_forge import main as run_cli


# Core execution entry points
@eidosian()
def main() -> int:
    """
    Command-line entry point for direct script execution.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.debug("🚀 Doc Forge main entry point invoked")
    return run_cli()


@eidosian()
def forge_docs(
    docs_dir: Optional[Union[str, Path]] = None,
    fix_toc: bool = True,
    fix_refs: bool = True,
    fix_syntax: bool = True,
    fix_duplicates: bool = True,
    validate: bool = True,
) -> bool:
    """
    One-shot function to update and fix documentation with Eidosian precision.

    Like a master blacksmith forging the perfect blade, this function applies
    multiple precise operations to shape your documentation into its ideal form.

    Args:
        docs_dir: Documentation directory (auto-detected if None)
        fix_toc: Whether to fix table of contents issues
        fix_refs: Whether to fix inline references
        fix_syntax: Whether to fix RST syntax issues
        fix_duplicates: Whether to fix duplicate object descriptions
        validate: Whether to validate documentation

    Returns:
        bool: True if all operations succeeded, False otherwise
    """
    # Resolve docs directory - find our forge
    if docs_dir is None:
        docs_dir = get_docs_dir()
    else:
        docs_dir = Path(docs_dir)

    repo_root = get_repo_root()
    logger.info(f"🔥 Forging documentation in {docs_dir}")

    success = True
    operations_performed = 0

    # Run requested operations - each strike of the hammer must be precise
    if fix_toc:
        logger.info("📚 Fixing table of contents structure")
        toc_result = update_toctrees(docs_dir)
        success = success and (toc_result >= 0)
        operations_performed += 1

    if fix_refs:
        logger.info("🔗 Fixing inline references")
        refs_result = fix_inline_references(docs_dir)
        success = success and (refs_result >= 0)

        logger.info("🧩 Resolving ambiguous cross-references")
        try:
            fix_ambiguous_references(repo_root)
            operations_performed += 2
        except Exception as e:
            logger.error(f"⚠️ Error fixing ambiguous references: {e}")
            success = False

    if fix_syntax:
        logger.info("📝 Polishing RST syntax")
        try:
            syntax_fixer = RstSyntaxPerfector(docs_dir)
            fixed_count = syntax_fixer.fix_inline_text_issues()
            logger.info(f"✓ Fixed RST syntax issues in {fixed_count} files")

            docstring_fixer = DocstringFixer(docs_dir)
            fixed_count = docstring_fixer.fix_all_files()
            logger.info(f"✓ Fixed docstring formatting in {fixed_count} files")
            operations_performed += 2
        except Exception as e:
            logger.error(f"⚠️ Error fixing syntax: {e}")
            success = False

    if fix_duplicates:
        logger.info("🧿 Resolving duplicate object descriptions")
        try:
            harmonizer = DuplicateObjectHarmonizer(docs_dir)
            fixed_count = harmonizer.fix_duplicate_objects()
            logger.info(f"✓ Harmonized {fixed_count} duplicate objects")
            operations_performed += 1
        except Exception as e:
            logger.error(f"⚠️ Error fixing duplicates: {e}")
            success = False

    if validate:
        logger.info("🔍 Validating documentation integrity")
        discrepancies = validate_docs(repo_root)
        if discrepancies:
            logger.warning(f"⚠️ Found {sum(len(v) for v in discrepancies.values())} documentation discrepancies")
            success = False
        else:
            logger.info("✓ Documentation validation passed")
        operations_performed += 1

    # Final report - the master craftsman's assessment
    if success:
        logger.info(f"✨ Documentation forging complete! Performed {operations_performed} operations successfully")
    else:
        logger.warning("⚠️ Documentation forging completed with issues")

    return success


# Convenient aliases - different tools for different crafters
update_docs = forge_docs  # For those who prefer literal naming
perfect_docs = forge_docs  # For the perfectionists among us


class DocForge:
    """
    Main DocForge class for documentation generation and management.

    Provides methods to generate README files and API documentation
    from project source code.
    """

    def __init__(self, docs_dir: Optional[Union[str, Path]] = None):
        """Initialize DocForge with optional documentation directory."""
        self.docs_dir = Path(docs_dir) if docs_dir else get_docs_dir()
        self.repo_root = get_repo_root()

    @eidosian()
    def generate_readme(self, info: dict) -> str:
        """
        Generate a README file from project info.

        Args:
            info: Dictionary with keys:
                - name: Project name
                - description: Project description
                - features: List of feature strings

        Returns:
            Markdown string for README
        """
        name = info.get("name", "Project")
        description = info.get("description", "")
        features = info.get("features", [])

        readme = f"# 🔮 {name}\n\n"
        readme += f"{description}\n\n"

        if features:
            readme += "## ✨ Features\n\n"
            for feature in features:
                readme += f"- {feature}\n"
            readme += "\n"

        return readme

    @eidosian()
    def extract_and_generate_api_docs(self, source_dir: Union[str, Path]) -> str:
        """
        Extract docstrings and generate API documentation.

        Args:
            source_dir: Directory containing Python source files

        Returns:
            Markdown string with API documentation
        """
        import ast

        source_dir = Path(source_dir)
        api_docs = "# API Reference\n\n"

        for py_file in source_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source)

                module_doc = ast.get_docstring(tree) or ""
                api_docs += f"## Module: `{py_file.name}`\n\n"
                if module_doc:
                    api_docs += f"{module_doc}\n\n"

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_doc = ast.get_docstring(node) or ""
                        args = [a.arg for a in node.args.args]
                        api_docs += f"### `def {node.name}({', '.join(args)})`\n\n"
                        if func_doc:
                            api_docs += f"{func_doc}\n\n"
                    elif isinstance(node, ast.ClassDef):
                        class_doc = ast.get_docstring(node) or ""
                        api_docs += f"### `class {node.name}`\n\n"
                        if class_doc:
                            api_docs += f"{class_doc}\n\n"

            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")
                continue

        return api_docs


# When imported as a module, show our banner
logger.debug(f"🌀 Doc Forge v{get_version_string()} loaded - Eidosian Documentation System")
