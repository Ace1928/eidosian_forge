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
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge")

# Import version info from the dedicated version module - our temporal anchor
from .version import VERSION as __version__
from .version import get_version_string

# Core path utilities - foundation of structure, the skeleton of our cathedral
from .utils.paths import get_repo_root, get_docs_dir

# Documentation management components - the specialized organs of our organism
from .update_toctrees import update_toctrees
from .fix_inline_refs import fix_inline_references
from .doc_validator import validate_docs

# Import more specialized fixers - the surgical instruments of our toolkit
from .fix_cross_refs import fix_ambiguous_references
# Import more specialized fixers - the surgical instruments of our toolkit
from .fix_cross_refs import fix_ambiguous_references
from .fix_rst_syntax import RstSyntaxPerfector
from .fix_docstrings import DocstringFixer
from .fix_duplicate_objects import DuplicateObjectHarmonizer
try:
    from .run import main as run_cli
except ImportError:
    from .doc_forge import main as run_cli

# Core execution entry points
def main() -> int:
    """
    Command-line entry point for direct script execution.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.debug("🚀 Doc Forge main entry point invoked")
    return run_cli()

def forge_docs(
    docs_dir: Optional[Union[str, Path]] = None, 
    fix_toc: bool = True, 
    fix_refs: bool = True, 
    fix_syntax: bool = True,
    fix_duplicates: bool = True,
    validate: bool = True
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

# When imported as a module, show our banner
logger.debug(f"🌀 Doc Forge v{get_version_string()} loaded - Eidosian Documentation System")
