#!/usr/bin/env python3
# 🌀 Eidosian Documentation Manifest Manager
"""
Documentation Manifest Manager - Single Source of Truth for Docs Architecture

This module manages the documentation manifest, ensuring perfect
consistency between documentation sources, structure, and metadata.
It implements a dynamic registry system that tracks all documentation 
sources and their relationships with Eidosian precision.

Following Eidosian principles of:
- Contextual Integrity: Every document mapped to its exact purpose
- Structure as Control: Perfect organization of documentation architecture
- Self-Awareness as Foundation: System that knows and adapts its own structure
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Union, Optional

# 📊 Self-aware logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("eidosian_docs.manifest_manager")

class DocManifestManager:
    """
    Documentation Manifest Manager - Central registry for documentation architecture.
    
    Like a master architect who knows every blueprint of the cathedral! 🏗️
    """
    
    def __init__(self, repo_root: Path):
        """
        Initialize the manifest manager with the repository root.
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = repo_root
        self.docs_dir = repo_root / "docs"
        self.manifest_path = self.docs_dir / "docs_manifest.json"
        self.manifest: Dict[str, Any] = {}
        self.doc_sources: Dict[str, Path] = {}
        self.doc_references: Dict[str, Set[str]] = {}
        
        # Load existing manifest if available
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load the manifest file with perfect precision."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                logger.info(f"📚 Loaded manifest from {self.manifest_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load manifest: {e}")
                # Initialize with default structure
                self._initialize_default_manifest()
        else:
            logger.warning(f"⚠️ Manifest not found at {self.manifest_path}")
            self._initialize_default_manifest()
    
    def _initialize_default_manifest(self) -> None:
        """Initialize a default manifest structure with Eidosian clarity."""
        self.manifest = {
            "version": "1.0.0",
            "project_name": Path(self.repo_root).name,
            "documentation_categories": {
                "manual": {
                    "title": "Manual Documentation",
                    "description": "Hand-crafted documentation by development team",
                    "sections": {}
                },
                "auto": {
                    "title": "Auto-generated Documentation",
                    "description": "Documentation generated from source code",
                    "sections": {}
                },
                "source": {
                    "title": "Source Documentation",
                    "description": "Documentation of the source code structure",
                    "sections": {}
                },
                "assets": {
                    "title": "Assets",
                    "description": "Documentation assets and resources",
                    "sections": {}
                }
            },
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "validation_status": {
                    "missing_docs": [],
                    "outdated_docs": [],
                    "orphaned_docs": []
                },
                "build_info": {
                    "last_build": "",
                    "build_status": ""
                }
            }
        }
        logger.info("📝 Initialized default manifest structure")
    
    def save_manifest(self) -> None:
        """Save the manifest with elegant precision."""
        try:
            # Update the last_updated timestamp
            self.manifest["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Ensure the directory exists
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with pretty formatting
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=4, sort_keys=False)
            
            logger.info(f"✅ Saved manifest to {self.manifest_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save manifest: {e}")
    
    def discover_documentation(self) -> None:
        """
        Discover all documentation sources and update the manifest.
        Like a cartographer mapping uncharted territory! 🧭
        """
        start_time = time.time()
        
        # Track discovered documents
        discovered_docs: Dict[str, Dict[str, Dict[str, Any]]] = {
            "manual": {},
            "auto": {},
            "source": {},
            "assets": {}
        }
        
        # Discover manual documentation
        manual_docs = self._discover_manual_docs()
        for category, docs in manual_docs.items():
            discovered_docs["manual"][category] = docs
        
        # Discover auto-generated documentation
        auto_docs = self._discover_auto_docs()
        for category, docs in auto_docs.items():
            discovered_docs["auto"][category] = docs
        
        # Update the manifest with discovered documentation
        self._update_manifest_with_discovered_docs(discovered_docs)
        
        # Update metadata
        self.manifest["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # Calculate execution time with precision
        execution_time = time.time() - start_time
        logger.info(f"🔍 Documentation discovery completed in {execution_time:.2f}s")
    
    def _discover_manual_docs(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover manually written documentation.
        Manual docs are the artisanal treasures of knowledge! 📜
        
        Returns:
            Dictionary mapping category names to document collections
        """
        manual_docs: Dict[str, Dict[str, Any]] = {}
        
        # Map of common manual documentation directories to categories
        manual_dirs = {
            "getting_started": self.docs_dir / "manual" / "getting_started",
            "guides": self.docs_dir / "manual" / "guides",
            "tutorials": self.docs_dir / "manual" / "tutorials",
            "concepts": self.docs_dir / "manual" / "concepts",
            "reference": self.docs_dir / "manual" / "reference",
            "faq": self.docs_dir / "manual" / "faq",
            "examples": self.docs_dir / "manual" / "examples"
        }
        
        # Check each directory for documentation files
        for category, directory in manual_dirs.items():
            if directory.exists():
                docs = self._scan_directory_for_docs(directory, "manual")
                if docs:
                    manual_docs[category] = {
                        "path": str(directory.relative_to(self.docs_dir)),
                        "title": category.replace("_", " ").title(),
                        "docs": docs
                    }
        
        return manual_docs
    
    def _discover_auto_docs(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover auto-generated documentation.
        Auto-docs are the mechanized precision of the documentation system! 🤖
        
        Returns:
            Dictionary mapping category names to document collections
        """
        auto_docs: Dict[str, Dict[str, Any]] = {}
        
        # Map of common auto-generated documentation directories to categories
        auto_dirs = {
            "api": self.docs_dir / "auto" / "api",
            "source": self.docs_dir / "auto" / "source",
            "modules": self.docs_dir / "auto" / "modules",
            "classes": self.docs_dir / "auto" / "classes",
            "functions": self.docs_dir / "auto" / "functions"
        }
        
        # Check each directory for documentation files
        for category, directory in auto_dirs.items():
            if directory.exists():
                docs = self._scan_directory_for_docs(directory, "auto")
                if docs:
                    auto_docs[category] = {
                        "path": str(directory.relative_to(self.docs_dir)),
                        "title": category.replace("_", " ").title(),
                        "docs": docs
                    }
        
        return auto_docs
    
    def _scan_directory_for_docs(self, directory: Path, doc_type: str) -> List[Dict[str, Any]]:
        """
        Scan a directory for documentation files.
        Every file has a story to tell, we just need to listen! 👂
        
        Args:
            directory: Directory to scan
            doc_type: Type of documentation ("manual" or "auto")
            
        Returns:
            List of document metadata
        """
        docs: List[Dict[str, Any]] = []
        
        # Skip if the directory doesn't exist
        if not directory.exists():
            return docs
        
        # Get all markdown and RST files
        md_files = list(directory.glob("**/*.md"))
        rst_files = list(directory.glob("**/*.rst"))
        
        for file_path in sorted(md_files + rst_files):
            # Skip files in underscore directories
            if any(part.startswith("_") for part in file_path.parts):
                continue
                
            # Extract document metadata
            doc_metadata = self._extract_doc_metadata(file_path, doc_type)
            docs.append(doc_metadata)
            
            # Register this document in our sources map
            rel_path = str(file_path.relative_to(self.docs_dir))
            self.doc_sources[rel_path] = file_path
        
        return docs
    
    def _extract_doc_metadata(self, file_path: Path, doc_type: str) -> Dict[str, Any]:
        """
        Extract metadata from a documentation file.
        Like a digital archaeologist uncovering hidden treasures! 🏺
        
        Args:
            file_path: Path to the documentation file
            doc_type: Type of documentation ("manual" or "auto")
            
        Returns:
            Document metadata
        """
        metadata: Dict[str, Any] = {
            "path": str(file_path.relative_to(self.docs_dir)),
            "title": file_path.stem.replace("_", " ").title(),
            "type": doc_type,
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        try:
            # Read the file to extract more metadata
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Extract title
            if file_path.suffix == ".md":
                # Look for markdown title
                title_match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
                if title_match:
                    metadata["title"] = title_match.group(1).strip()
            elif file_path.suffix == ".rst":
                # Look for RST title
                title_match = re.search(r'^(.*?)\n[=]+\s*$', content, re.MULTILINE)
                if title_match:
                    metadata["title"] = title_match.group(1).strip()
            
            # Extract description (first paragraph)
            desc_match = re.search(r'(?:^|#.*?\n\n)(.*?)(?:\n\n|\n#|\Z)', content, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()
                # Clean up and truncate description
                description = re.sub(r'\s+', ' ', description)
                metadata["description"] = description[:150] + ("..." if len(description) > 150 else "")
            
            # Extract references to other documents
            references: Set[str] = set()
            
            # Markdown links
            md_links = re.finditer(r'\[.*?\]\((.*?)\)', content)
            for match in md_links:
                link = match.group(1).strip()
                if not link.startswith(("http:", "https:", "#")):
                    references.add(link)
            
            # RST links
            rst_links = re.finditer(r':doc:`(.*?)`', content)
            for match in rst_links:
                link = match.group(1).strip()
                references.add(link)
            
            # Store references
            if references:
                # Store references as a list in the metadata dictionary
                metadata["references"] = list(references)
                # Also store in our internal references map
                self.doc_references[metadata["path"]] = references
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def _update_manifest_with_discovered_docs(self, discovered_docs: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Update the manifest with discovered documentation.
        Perfect synchronization between discovery and registry! 🔄
        
        Args:
            discovered_docs: Dictionary of discovered documentation
        """
        # Update manual documentation categories
        for category, docs in discovered_docs["manual"].items():
            if category not in self.manifest["documentation_categories"]["manual"]["sections"]:
                # Add new category
                self.manifest["documentation_categories"]["manual"]["sections"][category] = {
                    "path": docs["path"],
                    "title": docs["title"]
                }
        
        # Update auto-generated documentation categories
        for category, docs in discovered_docs["auto"].items():
            if category not in self.manifest["documentation_categories"]["auto"]["sections"]:
                # Add new category
                self.manifest["documentation_categories"]["auto"]["sections"][category] = {
                    "path": docs["path"],
                    "title": docs["title"]
                }
    
    def validate_documentation(self) -> Dict[str, List[str]]:
        """
        Validate documentation for completeness and consistency.
        Quality control with Eidosian precision! 🔍
        
        Returns:
            Dictionary with validation results
        """
        results: Dict[str, List[str]] = {
            "missing_docs": [],
            "outdated_docs": [],
            "orphaned_docs": []
        }
        
        # Check for missing documentation (required docs that don't exist)
        required_docs = self._get_required_docs()
        for doc_path in required_docs:
            full_path = self.docs_dir / doc_path
            if not full_path.exists():
                results["missing_docs"].append(doc_path)
        
        # Check for outdated documentation (docs that are older than their source)
        outdated_docs = self._check_for_outdated_docs()
        results["outdated_docs"] = outdated_docs
        
        # Check for orphaned documentation (docs not referenced anywhere)
        orphaned_docs = self._check_for_orphaned_docs()
        results["orphaned_docs"] = orphaned_docs
        
        # Update manifest validation status
        self.manifest["metadata"]["validation_status"] = results
        
        return results
    
    def _get_required_docs(self) -> List[str]:
        """
        Get a list of required documentation files.
        The essential pillars of the documentation temple! 🏛️
        
        Returns:
            List of required documentation paths
        """
        required_docs = [
            "manual/getting_started/index.md",
            "manual/getting_started/installation.md",
            "manual/getting_started/quickstart.md",
            "README.md"
        ]
        
        return required_docs
    
    def _check_for_outdated_docs(self) -> List[str]:
        """
        Check for outdated documentation files.
        Documentation should age like fine wine, not milk! 🍷
        
        Returns:
            List of outdated documentation paths
        """
        outdated_docs: List[str] = []
        
        # Add logic to check if docs are older than their source
        # This is project-specific and would typically compare
        # documentation timestamps with source code timestamps
        
        return outdated_docs
    
    def _check_for_orphaned_docs(self) -> List[str]:
        """
        Check for orphaned documentation files.
        No document deserves to be abandoned in the digital wilderness! 🏝️
        
        Returns:
            List of orphaned documentation paths
        """
        orphaned_docs: List[str] = []
        
        # Build a set of all referenced documents
        all_references: Set[str] = set()
        for references in self.doc_references.values():
            all_references.update(references)
        
        # Check for documents that aren't referenced
        for doc_path in self.doc_sources:
            # Skip index files and README
            if doc_path.endswith(("index.md", "index.rst", "README.md")):
                continue
                
            # Check if this document is referenced anywhere
            if doc_path not in all_references:
                orphaned_docs.append(doc_path)
        
        return orphaned_docs
    
    def update_build_info(self, status: str) -> None:
        """
        Update the build information in the manifest.
        Every build tells a story of progress! 📈
        
        Args:
            status: Build status ("success", "failed", "in_progress")
        """
        self.manifest["metadata"]["build_info"] = {
            "last_build": datetime.now().isoformat(),
            "build_status": status
        }
        self.save_manifest()
    
    def get_manifest_data(self) -> Dict[str, Any]:
        """
        Get the current manifest data.
        The source of truth for our documentation architecture! 📚
        
        Returns:
            Current manifest data
        """
        return self.manifest
    
    def generate_index(self) -> str:
        """
        Generate a documentation index based on the manifest.
        The map that guides users through the documentation landscape! 🗺️
        
        Returns:
            Index content as string
        """
        index_content = f"# {self.manifest['project_name']} Documentation\n\n"
        
        # Add an introduction
        index_content += "Welcome to the documentation for this project.\n\n"
        
        # Add manual documentation sections
        index_content += "## Manual Documentation\n\n"
        manual_sections = self.manifest["documentation_categories"]["manual"]["sections"]
        for _, section in manual_sections.items():
            index_content += f"### {section['title']}\n\n"
            index_content += f"[View {section['title']}]({section['path']}/index.md)\n\n"
        
        # Add auto-generated documentation sections
        index_content += "## Auto-Generated Documentation\n\n"
        auto_sections = self.manifest["documentation_categories"]["auto"]["sections"]
        for _, section in auto_sections.items():
            index_content += f"### {section['title']}\n\n"
            index_content += f"[View {section['title']}]({section['path']}/index.md)\n\n"
        
        return index_content
    
    def sync_manifest_with_filesystem(self) -> None:
        """
        Synchronize the manifest with the filesystem.
        Perfect harmony between digital reality and its representation! 🧘
        """
        # Discover all documentation
        self.discover_documentation()
        
        # Validate documentation
        self.validate_documentation()
        
        # Save the manifest
        self.save_manifest()

def load_doc_manifest(repo_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load and synchronize the documentation manifest with Eidosian precision.
    
    This function serves as the universal interface to the manifest system,
    ensuring the manifest is synchronized with the current documentation state.
    
    Args:
        repo_path: Path to the repository root (auto-detected if None)
        
    Returns:
        Dictionary with the current manifest data
    """
    # Auto-detect repository root if not provided
    if repo_path is None:
        # Try common locations
        possible_paths = [
            Path("."),
            Path(".."),
            Path(__file__).resolve().parent.parent.parent
        ]
        
        for path in possible_paths:
            if (path / ".git").exists() or (path / "docs").exists():
                repo_path = path
                break
        
    # Ensure repo_path is not None before proceeding - using different approach to avoid type error
    if repo_path is None:
        logger.error("❌ Repository path not specified and couldn't be auto-detected")
        return {"error": "Repository path not found"}
    
    repo_path_obj = Path(repo_path)
    
    logger.info(f"📄 Loading documentation manifest from {repo_path_obj}")
    
    # Create manifest manager
    manager = DocManifestManager(repo_path_obj)
    
    # Synchronize manifest with filesystem to ensure it's up-to-date
    manager.sync_manifest_with_filesystem()
    
    # Get the current manifest data
    manifest = manager.get_manifest_data()
    
    logger.info("✅ Documentation manifest loaded and synchronized")
    return manifest

# Command-line execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Eidosian Documentation Manifest Manager")
    parser.add_argument("--repo-root", type=Path, help="Repository root directory")
    parser.add_argument("--discover", action="store_true", help="Discover documentation")
    parser.add_argument("--validate", action="store_true", help="Validate documentation")
    parser.add_argument("--sync", action="store_true", help="Synchronize manifest with filesystem")
    parser.add_argument("--generate-index", action="store_true", help="Generate documentation index")
    args = parser.parse_args()
    
    # Determine repository root
    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    
    # Create manifest manager
    manager = DocManifestManager(repo_root)
    
    if args.discover:
        manager.discover_documentation()
        manager.save_manifest()
        print("✅ Documentation discovery complete")
    
    if args.validate:
        results = manager.validate_documentation()
        print(f"🔍 Validation results:")
        print(f"  - Missing docs: {len(results['missing_docs'])}")
        print(f"  - Outdated docs: {len(results['outdated_docs'])}")
        print(f"  - Orphaned docs: {len(results['orphaned_docs'])}")
        manager.save_manifest()
    
    if args.sync:
        manager.sync_manifest_with_filesystem()
        print("✅ Manifest synchronized with filesystem")
    
    if args.generate_index:
        index_content = manager.generate_index()
        index_path = repo_root / "docs" / "index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)
        print(f"✅ Generated index at {index_path}")
