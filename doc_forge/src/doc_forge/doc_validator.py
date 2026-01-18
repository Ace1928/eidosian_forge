#!/usr/bin/env python3
# 🌀 Eidosian Documentation Validator
"""
Documentation Validator - Ensuring Documentation Perfection

This module validates documentation for completeness, consistency, and correctness,
following Eidosian principles of precision, structure, and thoroughness.

The validator checks for:
- Missing required documentation
- Structural completeness and adherence to standards
- Consistency between source code and documentation
- Documentation quality and clarity
"""

import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple

# Import project-wide utilities
from .utils.paths import get_repo_root, get_docs_dir, resolve_path
from .source_discovery import discover_documentation, discover_code_structures

# 📊 Self-aware logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.doc_validator")

class DocumentationValidator:
    """Validates documentation with Eidosian precision and thoroughness."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize the documentation validator.
        
        Args:
            repo_root: Repository root directory
        """
        self.repo_root = repo_root
        self.docs_dir = get_docs_dir()
        
        # Store validation results
        self.discrepancies: Dict[str, List[str]] = {
            "missing_docs": [],
            "inconsistent_docs": [],
            "structural_issues": [],
            "quality_issues": []
        }
        
        # Cache discovery results for efficiency
        self.discovered_docs = None
        self.code_structures = None
        
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all aspects of documentation with Eidosian thoroughness.
        
        Returns:
            Dictionary of discrepancies by category
        """
        logger.info(f"🔍 Validating documentation in {self.repo_root}")
        
        # Discover documentation and code structures
        self._discover_content()
        
        # Run validation checks
        self._validate_required_docs()
        self._validate_structure()
        self._validate_consistency()
        self._validate_quality()
        
        # Summarize results
        total_discrepancies = sum(len(issues) for issues in self.discrepancies.values())
        if total_discrepancies > 0:
            logger.warning(f"⚠️ Found {total_discrepancies} documentation discrepancies")
            for category, issues in self.discrepancies.items():
                if issues:
                    logger.warning(f"⚠️ {category.replace('_', ' ').title()}: {len(issues)} issues")
        else:
            logger.info("✅ Documentation validation passed with no discrepancies")
            
        return self.discrepancies
        
    def _discover_content(self) -> None:
        """Discover documentation and code structures for validation."""
        # Discover documentation
        self.discovered_docs = discover_documentation(self.docs_dir)
        
        # Discover code structures
        src_dir = self.repo_root / "src"
        if src_dir.exists():
            self.code_structures = discover_code_structures(src_dir)
        else:
            self.code_structures = discover_code_structures(self.repo_root)
            
        logger.info(f"📚 Discovered {sum(len(docs) for docs in self.discovered_docs.values())} documentation files")
        logger.info(f"🧩 Discovered {len(self.code_structures)} code structures")
    
    def _validate_required_docs(self) -> None:
        """Validate that all required documentation exists."""
        # Check for critical documentation files that should exist
        required_docs = [
            (self.docs_dir / "index.md", "Main documentation index"),
            (self.docs_dir / "README.md", "Documentation README"),
            (self.repo_root / "README.md", "Project README")
        ]
        
        # Check for "Getting Started" section
        getting_started_dir = self.docs_dir / "getting_started"
        if getting_started_dir.exists():
            required_docs.extend([
                (getting_started_dir / "index.md", "Getting Started index"),
                (getting_started_dir / "installation.md", "Installation guide")
            ])
        else:
            # Alternative location in user_docs structure
            user_docs_getting_started = self.docs_dir / "user_docs" / "getting_started"
            if user_docs_getting_started.exists():
                required_docs.extend([
                    (user_docs_getting_started / "index.md", "Getting Started index"),
                    (user_docs_getting_started / "installation.md", "Installation guide")
                ])
            else:
                # Record missing getting started section
                self.discrepancies["missing_docs"].append("Missing 'Getting Started' section (should be at docs/getting_started/ or docs/user_docs/getting_started/)")
        
        # Check for API documentation if we have code structures
        if self.code_structures:
            api_doc_candidates = [
                self.docs_dir / "api" / "index.md",
                self.docs_dir / "reference" / "api.md",
                self.docs_dir / "reference" / "index.md",
                self.docs_dir / "user_docs" / "reference" / "api.md",
                self.docs_dir / "auto_docs" / "api" / "index.md"
            ]
            
            if not any(path.exists() for path in api_doc_candidates):
                self.discrepancies["missing_docs"].append("Missing API documentation for code structures")
        
        # Check for all required docs
        for path, description in required_docs:
            if not path.exists():
                self.discrepancies["missing_docs"].append(f"Missing {description} at {path.relative_to(self.repo_root)}")
    
    def _validate_structure(self) -> None:
        """Validate documentation structure for consistency and completeness."""
        # Check for common structural issues
        
        # 1. Missing indices in directories with multiple documents
        for category, docs in self.discovered_docs.items():
            # Group docs by directory
            dir_docs = {}
            for doc in docs:
                dir_path = doc.path.parent
                if dir_path not in dir_docs:
                    dir_docs[dir_path] = []
                dir_docs[dir_path].append(doc)
            
            # Check each directory for an index file
            for dir_path, doc_list in dir_docs.items():
                # Skip if only one file in directory
                if len(doc_list) <= 1:
                    continue
                    
                # Skip if this directory already has an index file
                index_found = any(doc.path.name.lower() in ["index.md", "index.rst"] for doc in doc_list)
                if not index_found:
                    rel_path = dir_path.relative_to(self.repo_root)
                    self.discrepancies["structural_issues"].append(f"Directory {rel_path} has multiple docs but no index file")
        
        # 2. Check for duplicate titles
        titles_seen = {}
        for category, docs in self.discovered_docs.items():
            for doc in docs:
                if doc.title in titles_seen:
                    # There's a duplicate title
                    prev_path = titles_seen[doc.title].path.relative_to(self.repo_root)
                    curr_path = doc.path.relative_to(self.repo_root)
                    self.discrepancies["structural_issues"].append(
                        f"Duplicate document title '{doc.title}' in {curr_path} and {prev_path}"
                    )
                else:
                    titles_seen[doc.title] = doc
        
        # 3. Check for standards compliance in directory structure
        standard_dirs = {
            "getting_started", "user_guide", "api", "reference", "guides", 
            "tutorials", "examples", "concepts", "faq", "contributing"
        }
        
        # List all directories in the docs folder
        all_dirs = [p for p in self.docs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
        for dir_path in all_dirs:
            # Skip standard structure directories
            if dir_path.name.lower() in standard_dirs or dir_path.name in ["user_docs", "auto_docs", "ai_docs", "assets"]:
                continue
                
            # Check if this is a standard directory in our doc structure
            is_standard = False
            for category, sections in [
                ("user_docs", ["getting_started", "guides", "concepts", "reference", "examples", "faq"]),
                ("auto_docs", ["api", "introspected", "extracted"]),
                ("ai_docs", ["generated", "enhanced", "integrated"])
            ]:
                category_dir = self.docs_dir / category
                if category_dir.exists() and dir_path.name in sections:
                    is_standard = True
                    break
                    
            if not is_standard:
                self.discrepancies["structural_issues"].append(
                    f"Non-standard documentation directory: {dir_path.relative_to(self.repo_root)}"
                )
    
    def _validate_consistency(self) -> None:
        """Validate consistency between code and documentation."""
        if not self.code_structures:
            logger.debug("No code structures found, skipping consistency validation")
            return
            
        # Check that important code structures are documented
        documented_items = set()
        
        # Check for API documentation mentions
        for category, docs in self.discovered_docs.items():
            for doc in docs:
                try:
                    with open(doc.path, "r", encoding="utf-8") as f:
                        content = f.read().lower()
                        
                    # For each code structure, check if it's mentioned in the documentation
                    for code_item in self.code_structures:
                        item_name = code_item["name"]
                        if item_name.lower() in content:
                            documented_items.add(item_name)
                except Exception:
                    # Skip files we can't read
                    pass
        
        # Find undocumented public classes and functions
        for code_item in self.code_structures:
            item_name = code_item["name"]
            item_type = code_item["type"]
            
            # Skip private items (already filtered in discovery)
            if item_name.startswith("_"):
                continue
                
            # Skip common utility functions
            if item_name.lower() in ["main", "run", "setup", "test"]:
                continue
                
            # Skip if already documented
            if item_name in documented_items:
                continue
                
            self.discrepancies["inconsistent_docs"].append(
                f"Undocumented {item_type}: {item_name} in {code_item['file']}"
            )
    
    def _validate_quality(self) -> None:
        """Validate documentation quality standards."""
        # Check README.md exists and has essential sections
        readme_path = self.repo_root / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    
                # Check for essential README sections
                required_sections = {
                    "installation": ["installation", "getting started", "quick start", "setup"],
                    "usage": ["usage", "how to use", "examples", "getting started"],
                    "features": ["features", "what it does", "capabilities"]
                }
                
                for section, alternatives in required_sections.items():
                    if not any(alt in content for alt in alternatives):
                        self.discrepancies["quality_issues"].append(
                            f"README.md missing '{section}' section"
                        )
            except Exception:
                self.discrepancies["quality_issues"].append("Could not read README.md for quality check")
                
        # Check documentation files for quality issues
        quality_checks = 0
        quality_issues = 0
        
        for category, docs in self.discovered_docs.items():
            for doc in docs:
                try:
                    with open(doc.path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    quality_checks += 1
                    
                    # Check for very short documents (potentially incomplete)
                    if len(content.strip()) < 100:
                        rel_path = doc.path.relative_to(self.repo_root)
                        self.discrepancies["quality_issues"].append(
                            f"Document too short (potentially incomplete): {rel_path}"
                        )
                        quality_issues += 1
                        
                    # Check for headings structure (at least one heading)
                    if doc.path.suffix == ".md":
                        if not re.search(r'^#\s+', content, re.MULTILINE):
                            rel_path = doc.path.relative_to(self.repo_root)
                            self.discrepancies["quality_issues"].append(
                                f"Document missing headings: {rel_path}"
                            )
                            quality_issues += 1
                    elif doc.path.suffix == ".rst":
                        if not re.search(r'^[^\n]+\n[=\-~]+\s*$', content, re.MULTILINE):
                            rel_path = doc.path.relative_to(self.repo_root)
                            self.discrepancies["quality_issues"].append(
                                f"Document missing headings: {rel_path}"
                            )
                            quality_issues += 1
                            
                except Exception:
                    # Skip files we can't read
                    pass
        
        if quality_checks > 0:
            logger.debug(f"Quality validation: Checked {quality_checks} documents, found {quality_issues} issues")

def validate_docs(repo_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Validate documentation for completeness, consistency, and correctness.
    
    This function serves as the universal interface to the documentation validation system,
    ensuring all documentation meets Eidosian standards of precision and clarity.
    
    Args:
        repo_path: Repository path (auto-detected if None)
        
    Returns:
        Dictionary of discrepancies by category
    """
    # Auto-detect repository path if not provided
    if repo_path is None:
        try:
            repo_path = get_repo_root()
        except Exception as e:
            logger.error(f"❌ Failed to auto-detect repository path: {e}")
            return {"errors": [f"Failed to auto-detect repository path: {e}"]}
    
    repo_path = Path(repo_path)
    
    if not repo_path.is_dir():
        logger.error(f"❌ Repository path not found or not a directory: {repo_path}")
        return {"errors": [f"Repository path not found: {repo_path}"]}
    
    try:
        # Create validator and run validation
        validator = DocumentationValidator(repo_path)
        discrepancies = validator.validate_all()
        
        # Log a summary of validation results
        total_discrepancies = sum(len(issues) for issues in discrepancies.values())
        if total_discrepancies > 0:
            logger.warning(f"⚠️ Documentation validation found {total_discrepancies} issues")
        else:
            logger.info("✅ Documentation validation passed with no discrepancies")
            
        return discrepancies
        
    except Exception as e:
        logger.error(f"❌ Documentation validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"errors": [f"Validation failed with exception: {e}"]}

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate documentation for completeness and consistency.")
    parser.add_argument("repo_path", nargs="?", type=Path, help="Repository path")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    discrepancies = validate_docs(args.repo_path)
    
    if args.json:
        print(json.dumps(discrepancies, indent=2))
    else:
        total = sum(len(issues) for issues in discrepancies.values())
        if total > 0:
            print(f"⚠️ Found {total} documentation discrepancies:")
            for category, issues in discrepancies.items():
                if issues:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for issue in issues:
                        print(f"  - {issue}")
            sys.exit(1)
        else:
            print("✅ Documentation validation passed with no discrepancies")
            sys.exit(0)
