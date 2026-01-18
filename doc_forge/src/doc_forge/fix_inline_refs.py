#!/usr/bin/env python3
# 🌀 Eidosian Inline References Fixer
"""
Inline References Fixer - Perfect Link Connectivity

This module fixes inline references in documentation files, ensuring
perfect connectivity between documents and eliminating broken links.
Following Eidosian principles of precision and thoroughness.
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Pattern

# Import project-wide utilities
from .utils.paths import get_repo_root, get_docs_dir, resolve_path

# 📊 Self-aware logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.fix_inline_refs")

class InlineReferenceFixer:
    """Fixes inline references in documentation with Eidosian precision."""
    
    def __init__(self, docs_dir: Path):
        """
        Initialize the inline reference fixer.
        
        Args:
            docs_dir: Documentation directory root
        """
        self.docs_dir = docs_dir
        self.files_fixed = 0
        self.refs_fixed = 0
        
        # Patterns for different types of references
        self.patterns = {
            "markdown_links": re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            "markdown_refs": re.compile(r'\[([^\]]+)\]\[([^\]]+)\]'),
            "rst_refs": re.compile(r':(?:doc|ref):`([^`]+)`'),
            "html_links": re.compile(r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>')
        }
        
    def fix_all_files(self) -> int:
        """
        Fix inline references in all documentation files.
        
        Returns:
            Number of files fixed
        """
        logger.info(f"🔍 Scanning for documentation files in {self.docs_dir}")
        
        # Find all markdown and RST files
        md_files = list(self.docs_dir.glob("**/*.md"))
        rst_files = list(self.docs_dir.glob("**/*.rst"))
        all_files = md_files + rst_files
        
        logger.info(f"📚 Found {len(all_files)} documentation files ({len(md_files)} MD, {len(rst_files)} RST)")
        
        # Create a mapping of document paths
        path_mapping = self._create_path_mapping(all_files)
        
        # Process each file
        for file_path in all_files:
            self._fix_file_references(file_path, path_mapping)
            
        logger.info(f"✅ Fixed {self.refs_fixed} references in {self.files_fixed} files")
        return self.files_fixed
    
    def _create_path_mapping(self, all_files: List[Path]) -> Dict[str, Path]:
        """
        Create a mapping of document references to actual paths.
        
        Args:
            all_files: List of documentation file paths
            
        Returns:
            Dictionary mapping reference keys to actual paths
        """
        mapping = {}
        
        for file_path in all_files:
            # Get relative path from docs directory
            rel_path = file_path.relative_to(self.docs_dir)
            
            # Create different key variants for mapping:
            
            # 1. Full path with extension
            mapping[str(rel_path)] = file_path
            
            # 2. Full path without extension
            stem_path = rel_path.with_suffix("")
            mapping[str(stem_path)] = file_path
            
            # 3. Path with HTML extension (common in output references)
            html_path = rel_path.with_suffix(".html")
            mapping[str(html_path)] = file_path
            
            # 4. Filename only (for simple references)
            mapping[file_path.stem] = file_path
            
            # 5. Title-based reference (filename with underscores replaced by spaces)
            title_ref = file_path.stem.replace("_", " ")
            mapping[title_ref] = file_path
            
        return mapping
    
    def _fix_file_references(self, file_path: Path, path_mapping: Dict[str, Path]) -> bool:
        """
        Fix references in a single file.
        
        Args:
            file_path: Path to the file to fix
            path_mapping: Mapping of reference keys to actual paths
            
        Returns:
            True if file was modified, False otherwise
        """
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Make a copy of the original content
            original_content = content
            
            # Determine file type
            is_markdown = file_path.suffix.lower() == ".md"
            is_rst = file_path.suffix.lower() == ".rst"
            
            # Fix references based on file type
            if is_markdown:
                content = self._fix_markdown_refs(content, file_path, path_mapping)
            elif is_rst:
                content = self._fix_rst_refs(content, file_path, path_mapping)
            
            # Write back if changed
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                self.files_fixed += 1
                logger.debug(f"🔧 Fixed references in {file_path.relative_to(self.docs_dir)}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Error fixing references in {file_path}: {e}")
            return False
    
    def _fix_markdown_refs(self, content: str, file_path: Path, path_mapping: Dict[str, Path]) -> str:
        """
        Fix references in Markdown content.
        
        Args:
            content: File content
            file_path: Path to the file
            path_mapping: Mapping of reference keys to actual paths
            
        Returns:
            Updated content
        """
        # Track original content for comparison
        original_content = content
        file_dir = file_path.parent
        
        # Fix markdown links [text](link)
        def fix_md_link(match: re.Match) -> str:
            text = match.group(1)
            link = match.group(2)
            
            # Skip external links and anchors
            if link.startswith(("http:", "https:", "mailto:", "#", "/")):
                return match.group(0)
            
            # Try to resolve the link path
            link_path = self._resolve_link_path(link, file_dir, path_mapping)
            if link_path:
                # Create a relative path from the current file to the target
                rel_path = os.path.relpath(link_path, file_dir)
                rel_path = rel_path.replace("\\", "/")  # Normalize path separators
                
                # Use .md extension for links in Markdown files
                if rel_path.endswith((".rst", ".html")):
                    rel_path = rel_path[:-len(rel_path.split(".")[-1])-1] + ".md"
                    
                self.refs_fixed += 1
                return f"[{text}]({rel_path})"
            
            return match.group(0)
            
        content = self.patterns["markdown_links"].sub(fix_md_link, content)
        
        # Fix markdown reference links [text][ref]
        # These are more complex and would need a second pass to fix the reference definitions
        
        return content
    
    def _fix_rst_refs(self, content: str, file_path: Path, path_mapping: Dict[str, Path]) -> str:
        """
        Fix references in reStructuredText content.
        
        Args:
            content: File content
            file_path: Path to the file
            path_mapping: Mapping of reference keys to actual paths
            
        Returns:
            Updated content
        """
        # Track original content for comparison
        original_content = content
        file_dir = file_path.parent
        
        # Fix RST doc references :doc:`link`
        def fix_rst_doc_ref(match: re.Match) -> str:
            link = match.group(1)
            
            # Extract the link text if present
            link_text = None
            if " <" in link and ">" in link:
                parts = link.split(" <", 1)
                link_text = parts[0]
                link = parts[1].rstrip(">")
            
            # Try to resolve the link path
            link_path = self._resolve_link_path(link, file_dir, path_mapping)
            if link_path:
                # Create a relative path from the docs root to the target
                rel_path = link_path.relative_to(self.docs_dir)
                rel_path = str(rel_path.with_suffix("")).replace("\\", "/")
                
                self.refs_fixed += 1
                if link_text:
                    return f":doc:`{link_text} <{rel_path}>`"
                else:
                    return f":doc:`{rel_path}`"
            
            return match.group(0)
            
        content = self.patterns["rst_refs"].sub(fix_rst_doc_ref, content)
        
        return content
    
    def _resolve_link_path(self, link: str, file_dir: Path, path_mapping: Dict[str, Path]) -> Optional[Path]:
        """
        Resolve a link to an actual file path.
        
        Args:
            link: Link to resolve
            file_dir: Directory of the current file
            path_mapping: Mapping of reference keys to actual paths
            
        Returns:
            Resolved path or None if not found
        """
        # Remove any anchor part from the link
        link = link.split("#")[0]
        
        # Try direct lookup in path mapping
        if link in path_mapping:
            return path_mapping[link]
        
        # Try with various extensions
        for ext in ["", ".md", ".rst", ".html"]:
            if link + ext in path_mapping:
                return path_mapping[link + ext]
        
        # Try relative to current file
        rel_path = file_dir / link
        if rel_path.exists():
            return rel_path
        
        # Try with various extensions relative to current file
        for ext in [".md", ".rst", ".html"]:
            if (file_dir / (link + ext)).exists():
                return file_dir / (link + ext)
        
        # Try if it's a directory with an index file
        index_paths = [
            file_dir / link / "index.md",
            file_dir / link / "index.rst",
            self.docs_dir / link / "index.md",
            self.docs_dir / link / "index.rst"
        ]
        for index_path in index_paths:
            if index_path.exists():
                return index_path
        
        return None

def fix_inline_references(docs_dir: Optional[Path] = None) -> int:
    """
    Fix inline references in all documentation files.
    
    This function serves as the universal interface to the reference fixing system,
    ensuring all documentation links are correct and functional.
    
    Args:
        docs_dir: Documentation directory (auto-detected if None)
        
    Returns:
        Number of files fixed (negative if there was an error)
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
    
    logger.info(f"🔍 Fixing inline references in {docs_dir}")
    
    try:
        # Create fixer and run
        fixer = InlineReferenceFixer(docs_dir)
        files_fixed = fixer.fix_all_files()
        
        logger.info(f"✅ Reference fixing complete. Fixed {files_fixed} files")
        return files_fixed
    except Exception as e:
        logger.error(f"❌ Reference fixing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return -1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix inline references in documentation.")
    parser.add_argument("docs_dir", nargs="?", type=Path, help="Documentation directory")
    args = parser.parse_args()
    
    result = fix_inline_references(args.docs_dir)
    sys.exit(0 if result >= 0 else 1)
