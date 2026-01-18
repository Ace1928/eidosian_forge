#!/usr/bin/env python3
# 🌀 Eidosian TOC Tree Management
"""
TOC Tree Management - Perfect Documentation Hierarchy

This module analyzes and updates table of contents trees across documentation,
ensuring perfect organization, navigation, and completeness following Eidosian
principles.

Following Eidosian principles of:
- Structure as Control: Perfect organization of documentation hierarchy
- Flow Like a River: Seamless navigation between documentation sections
- Self-Awareness: Understanding the document ecosystem
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

# Import project-wide utilities
from .utils.paths import get_repo_root, get_docs_dir, resolve_path
from .source_discovery import DocumentationDiscovery, discover_documentation

# 📊 Self-Aware Logging - The foundation of understanding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("doc_forge.toctrees")

class TocTreeManager:
    """
    Table of Contents Tree Manager with perfect Eidosian structure awareness.
    
    Maintains and updates documentation hierarchy for optimal organization.
    """
    
    def __init__(self, docs_dir: Path):
        """
        Initialize the TOC tree manager.
        
        Args:
            docs_dir: Root directory of documentation
        """
        self.docs_dir = docs_dir
        self.files_updated = 0
        self.toc_entries_added = 0
        self.discovery = DocumentationDiscovery(docs_dir=docs_dir)
        self.documents = self.discovery.discover_all()
        self.toc_structure = self.discovery.generate_toc_structure(self.documents)
    
    def update_all_toctrees(self) -> int:
        """
        Update all table of contents trees with Eidosian precision.
        
        Returns:
            Number of files updated
        """
        logger.info("🔄 Updating table of contents trees")
        
        # Start with the main index file
        self._update_main_index()
        
        # Update section index files
        self._update_section_indices()
        
        # Create missing index files for sections that need them
        self._create_missing_indices()
        
        logger.info(f"✅ Updated {self.files_updated} files with {self.toc_entries_added} TOC entries")
        return self.files_updated
    
    def _update_main_index(self) -> None:
        """Update the main index.md file with the master table of contents."""
        index_path = self.docs_dir / "index.md"
        if not index_path.exists():
            logger.info("📝 Creating main index.md file")
            self._create_main_index()
            return
            
        logger.info("🔄 Updating main index.md file")
        
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check if there's already a TOC
        toc_match = re.search(r'```{toctree}.*?```', content, re.DOTALL)
        
        # Create new TOC content
        toc_content = self._generate_main_toc()
        
        if toc_match:
            # Replace existing TOC
            new_content = content[:toc_match.start()] + toc_content + content[toc_match.end():]
        else:
            # Add TOC after the first heading
            heading_match = re.search(r'^#\s+.*?$', content, re.MULTILINE)
            if heading_match:
                insert_pos = heading_match.end()
                new_content = content[:insert_pos] + "\n\n" + toc_content + "\n\n" + content[insert_pos:].lstrip()
            else:
                # No heading found, just append it
                new_content = content + "\n\n" + toc_content
        
        # Write the updated content
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        self.files_updated += 1
        logger.info("✅ Updated main index.md")
    
    def _create_main_index(self) -> None:
        """Create a new main index.md file with the master table of contents."""
        index_path = self.docs_dir / "index.md"
        
        # Get project name from directory name
        project_name = self.docs_dir.parent.name.replace("_", " ").title()
        
        content = f"# {project_name} Documentation\n\n"
        content += "Welcome to the documentation!\n\n"
        content += self._generate_main_toc()
        
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        self.files_updated += 1
        logger.info("✅ Created main index.md")
    
    def _generate_main_toc(self) -> str:
        """Generate the main table of contents for the index file."""
        toc = "```{toctree}\n"
        toc += ":maxdepth: 2\n"
        toc += ":caption: Contents\n\n"
        
        # Add section index files to the main TOC
        for section_name, section_data in self.toc_structure.items():
            if section_data["items"]:
                section_title = section_data["title"]
                toc += f"{section_name}/index\n"
        
        toc += "```\n"
        return toc
    
    def _update_section_indices(self) -> None:
        """Update or create section index files with appropriate TOC trees."""
        for section_name, section_data in self.toc_structure.items():
            if not section_data["items"]:
                # Skip empty sections
                continue
                
            # Ensure section directory exists
            section_dir = self.docs_dir / section_name
            section_dir.mkdir(exist_ok=True, parents=True)
            
            # Create or update section index
            index_path = section_dir / "index.md"
            self._update_section_index(section_name, section_data, index_path)
    
    def _update_section_index(self, section_name: str, section_data: Dict, index_path: Path) -> None:
        """Update a specific section index file."""
        # Check if index file exists
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Check if there's already a TOC
            toc_match = re.search(r'```{toctree}.*?```', content, re.DOTALL)
            
            # Create new TOC content
            toc_content = self._generate_section_toc(section_data)
            
            if toc_match:
                # Replace existing TOC
                new_content = content[:toc_match.start()] + toc_content + content[toc_match.end():]
            else:
                # Add TOC after the first heading
                heading_match = re.search(r'^#\s+.*?$', content, re.MULTILINE)
                if heading_match:
                    insert_pos = heading_match.end()
                    new_content = content[:insert_pos] + "\n\n" + toc_content + "\n\n" + content[insert_pos:].lstrip()
                else:
                    # No heading found, just append it
                    new_content = content + "\n\n" + toc_content
        else:
            # Create new index file
            title = section_data["title"]
            new_content = f"# {title}\n\n"
            new_content += self._generate_section_toc(section_data)
        
        # Write the updated content
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        self.files_updated += 1
        self.toc_entries_added += len(section_data["items"])
        logger.info(f"✅ Updated section index for {section_name}")
    
    def _generate_section_toc(self, section_data: Dict) -> str:
        """Generate the table of contents for a specific section."""
        toc = "```{toctree}\n"
        toc += ":maxdepth: 2\n"
        toc += f":caption: {section_data['title']}\n\n"
        
        # Add items to the section TOC
        for item in section_data["items"]:
            # Extract the path without extension
            url = item["url"]
            if url.endswith(".html"):
                url = url[:-5]  # Remove .html extension
                
            # Remove leading slash if present
            if url.startswith("/"):
                url = url[1:]
                
            # Ensure path is relative to section directory
            parts = url.split("/")
            if len(parts) > 1:
                # This is a path with subdirectories
                section = parts[0]
                if section == section_data["title"].lower().replace(" ", "_"):
                    # Remove redundant section prefix
                    url = "/".join(parts[1:])
            
            # Add to TOC
            toc += f"{url}\n"
        
        toc += "```\n"
        return toc
    
    def _create_missing_indices(self) -> None:
        """Create index files for directories that don't have them."""
        # Find all directories without index files
        for category, docs in self.documents.items():
            # Group docs by directory
            dir_docs = defaultdict(list)
            for doc in docs:
                dir_path = doc.path.parent
                dir_docs[dir_path].append(doc)
            
            # Check each directory for an index file
            for dir_path, dir_doc_list in dir_docs.items():
                # Skip if this is a top-level directory (handled by _update_section_indices)
                if dir_path == self.docs_dir:
                    continue
                    
                # Skip if this directory already has an index file
                index_md = dir_path / "index.md"
                index_rst = dir_path / "index.rst"
                if index_md.exists() or index_rst.exists():
                    continue
                    
                # Create a new index file
                dir_name = dir_path.name.replace("_", " ").title()
                content = f"# {dir_name}\n\n"
                
                # Add all documents in this directory to the TOC
                content += "```{toctree}\n"
                content += ":maxdepth: 1\n"
                content += f":caption: {dir_name}\n\n"
                
                for doc in dir_doc_list:
                    # Skip if this is an index file
                    if doc.path.stem.lower() == "index":
                        continue
                        
                    # Add to TOC
                    content += f"{doc.path.stem}\n"
                
                content += "```\n"
                
                # Write the index file
                with open(index_md, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                self.files_updated += 1
                self.toc_entries_added += len(dir_doc_list)
                logger.info(f"✅ Created index for {dir_path.relative_to(self.docs_dir)}")

def update_toctrees(docs_dir: Optional[Path] = None) -> int:
    """
    Update all table of contents trees across the documentation.
    
    This function serves as the universal interface to the TOC tree management system,
    ensuring all documentation is properly organized and navigable.
    
    Args:
        docs_dir: Documentation directory (auto-detected if None)
        
    Returns:
        Number of files updated (negative if there was an error)
    """
    # Auto-detect docs directory if not provided
    if docs_dir is None:
        try:
            docs_dir = get_docs_dir()
        except Exception as e:
            logger.error(f"❌ Failed to auto-detect docs directory: {e}")
            return -1
    
    docs_dir = Path(docs_dir)
    
    if not docs_dir.is_dir():
        logger.error(f"❌ Documentation directory not found or not a directory: {docs_dir}")
        return -1
    
    logger.info(f"🔍 Updating TOC trees in {docs_dir}")
    
    try:
        # Create manager and update TOC trees
        manager = TocTreeManager(docs_dir)
        files_updated = manager.update_all_toctrees()
        
        logger.info(f"✅ TOC tree update complete. Updated {files_updated} files")
        return files_updated
    except Exception as e:
        logger.error(f"❌ TOC tree update failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return -1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update documentation TOC trees.")
    parser.add_argument("docs_dir", nargs="?", type=Path, help="Documentation directory")
    args = parser.parse_args()
    
    result = update_toctrees(args.docs_dir)
    sys.exit(0 if result >= 0 else 1)