#!/usr/bin/env python3
# 🌀 Eidosian Documentation Fixer Runner

import sys
import logging
from pathlib import Path
from fix_docstrings import fix_fallback_context_docstring
from add_noindex import add_noindex_directives, fix_inline_literal_references
from fix_inline_refs import fix_inline_references

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("eidosian_docs.runner")

def run_all_fixers(docs_dir: Path = Path("../docs")) -> None:
    """Run all documentation fixers in sequence."""
    logger.info("🚀 Running Eidosian documentation fixers...")
    
    # Fix client.py fallback_context docstring
    fixed_docstring = fix_fallback_context_docstring()
    if fixed_docstring:
        logger.info("✅ Fixed fallback_context docstring")
    
    # Fix duplicate object descriptions - highest priority
    fixed_dups = add_noindex_directives(docs_dir)
    logger.info(f"✅ Fixed {fixed_dups} duplicate object descriptions")
    
    # Fix inline literals for better readability
    fixed_literals = fix_inline_literal_references(docs_dir)
    logger.info(f"✅ Fixed inline literals in {fixed_literals} files")
    
    # Fix inline references
    fixed_refs = fix_inline_references(docs_dir)
    logger.info(f"✅ Fixed inline references in {fixed_refs} files")
    
    # Run an additional pass on exceptions directory specifically
    exceptions_dir = docs_dir / "autoapi" / "ollama_forge" / "exceptions"
    if exceptions_dir.exists():
        try:
            from fix_rst_syntax import RstSyntaxPerfector
            perfector = RstSyntaxPerfector(docs_dir)
            fixed = perfector.fix_inline_text_issues(exceptions_dir)
            logger.info(f"✅ Fixed RST syntax issues in {fixed} exception files")
        except ImportError:
            logger.info("⚠️ RST syntax perfector not available - skipping exceptions fix")
    
    logger.info("✨ All fixers completed successfully")

if __name__ == "__main__":
    docs_dir = Path("../docs")
    if len(sys.argv) > 1:
        docs_dir = Path(sys.argv[1])
    
    run_all_fixers(docs_dir)
    print("\n💡 TIP: Run 'python -m sphinx -T -W --keep-going -b html docs docs/_build/html' to verify fixes")
