#!/usr/bin/env python3
# 🌀 Eidosian Docstring Fixer
"""
Docstring Fixer - Repairing Sphinx Formatting Issues

This script analyzes and fixes common docstring formatting issues in
auto-generated documentation files. It addresses indentation problems,
unmatched literal strings, and other formatting issues that cause
Sphinx build warnings or errors.

Following Eidosian principles of:
- Precision as Style: Clean, correctly formatted documentation
- Structure as Control: Properly structured documentation elements
- Recursive Refinement: Systematic cleanup and enhancement
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from functools import lru_cache

# 📊 Structured Logging - Self-Awareness Foundation
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger("eidosian_docs.docstring_fixer")

class DocstringFixer:
    """Fixes common docstring formatting issues with surgical precision. 🔧🔬"""
    
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir  # 📁 Documentation home
        self.autoapi_dir = docs_dir / "autoapi"  # 🤖 AutoAPI output
        self.fixed_count = 0  # 🧮 Success counter
        self.fix_patterns = {  # 🔍 Mapping of issues to fixes
            'unclosed_backtick': re.compile(r'`([^`\n]+?)(?=\n|\s\s|\))'),
            'unexpected_indent': re.compile(r'(^\s+)(\S.*?)\n\s{4,}(\S)'),
            'block_quote_spacing': re.compile(r'(\n\s+[^\s].*\n)([^\s])'),
            'example_colon': re.compile(r'((?:Example|Examples|Usage):)(\S)'),
        }
    
    def fix_all_files(self) -> int:
        """
        Fix all autoapi-generated documentation files with ruthless efficiency.
        
        Returns:
            Number of files fixed (victory count! 🏆)
        """
        if not self.autoapi_dir.exists():
            logger.warning(f"⚠️ AutoAPI directory not found at {self.autoapi_dir}")
            return 0
            
        # Process all RST files in autoapi directory - leave no stone unturned!
        for rst_file in self.autoapi_dir.glob("**/*.rst"):
            try:
                if self.fix_file(rst_file):
                    self.fixed_count += 1
            except Exception as e:
                logger.error(f"💥 Error processing {rst_file}: {e}")
                
        logger.info(f"✅ Fixed formatting issues in {self.fixed_count} files")
        return self.fixed_count
    
    @lru_cache(maxsize=8)  # 🚀 Speed up repeated pattern matches
    def _get_common_issues(self) -> List[re.Pattern]:
        """Return common patterns to fix - cached for velocity."""
        return list(self.fix_patterns.values())
    
    def fix_file(self, file_path: Path) -> bool:
        """
        Fix formatting issues in a single file with meticulous attention.
        
        Args:
            file_path: Path to the file to fix
            
        Returns:
            True if file was modified, False otherwise
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Apply each fix like a cascading waterfall 🌊
        updated_content = content
        updated_content = self.fix_inline_literals(updated_content)
        updated_content = self.fix_unexpected_indentation(updated_content)
        updated_content = self.fix_block_quotes(updated_content)
        updated_content = self.fix_code_block_formatting(updated_content)
        updated_content = self.add_no_index_to_duplicates(updated_content)
        
        # Write back if changed - don't fix what isn't broken
        if updated_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            logger.info(f"✨ Fixed formatting issues in {file_path.name}")
            return True
        return False
        
    def fix_inline_literals(self, content: str) -> str:
        """
        Fix unmatched inline literals like `text without closing backtick.
        A single unclosed backtick can unravel the fabric of documentation! 🧵
        """
        # Pattern to find unmatched backticks 
        pattern = r'`([^`\n]+?)(?=\n|\s\s|\))'
        
        # First fix unclosed backticks
        def replacement(match):
            text = match.group(1)
            # If already ends with backtick, leave it alone
            if text.endswith('`'):
                return match.group(0)
            return f"`{text}`"
            
        content = re.sub(pattern, replacement, content)
        
        # Then fix mismatched backtick pairs that could cause issues
        lines = content.split('\n')
        for i in range(len(lines)):
            # Count backticks in line - every opening needs a closing
            count = lines[i].count('`')
            if count % 2 == 1:  # Odd number of backticks - likely an issue
                # Try to find the simplest fix, which is often to add a backtick
                # at the end of an inline code reference
                pattern = r'`([^`]+)(\s*)$'
                match = re.search(pattern, lines[i])
                if match:
                    lines[i] = re.sub(pattern, r'`\1`\2', lines[i])
        
        return '\n'.join(lines)
    
    def fix_unexpected_indentation(self, content: str) -> str:
        """
        Fix unexpected indentation in docstrings. 
        Consistent indentation is the foundation of clarity! 📐
        """
        # Fix common pattern: Example code blocks with incorrect indentation
        lines = content.split('\n')
        in_example = False
        in_literal_block = False
        result = []
        
        for i, line in enumerate(lines):
            # Check if line starts an example - beginning of wisdom
            if any(marker in line for marker in ["Example:", "Examples:", "Usage:"]) and ":" in line:
                in_example = True
                result.append(line)
                # Add a blank line and properly indent the first line of code
                if i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith('..'):
                    result.append("")  # Add blank line before code
                continue
                
            # Check for literal block indicators - the signposts of code
            if line.strip() == "::":
                in_literal_block = True
                result.append(line)
                if i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith('..'):
                    result.append("")  # Add blank line before code block
                continue
                
            # Fix indentation in examples and literal blocks
            if (in_example or in_literal_block) and line.strip():
                # If we see a normal text line, exit example mode
                if not line.strip().startswith('`') and not line.startswith('    '):
                    in_example = False
                    in_literal_block = False
                    result.append(line)
                    continue
                    
                # Ensure code blocks have proper indentation - the cornerstone of structure
                if line.startswith('   ') and not line.startswith('    '):
                    # Fix inconsistent indentation (3 spaces instead of 4)
                    result.append(' ' + line)
                else:
                    result.append(line)
                continue
                
            # Check for the end of literal blocks
            if in_literal_block and not line.strip():
                result.append(line)
                if i+1 < len(lines) and not lines[i+1].startswith('    '):
                    in_literal_block = False
                continue
                
            result.append(line)
            
        return '\n'.join(result)
    
    def fix_block_quotes(self, content: str) -> str:
        """
        Fix block quote formatting issues with elegant precision.
        Block quotes need breathing room! 💨
        """
        # Fix common pattern: Block quote without blank line after
        pattern = r'(\n\s+[^\s].*\n)([^\s])'
        result = re.sub(pattern, r'\1\n\2', content)
        
        # Ensure blank lines before and after literal blocks
        pattern = r'([^\n])::\n(\S)'
        result = re.sub(pattern, r'\1::\n\n\2', result)
        
        return result
    
    def fix_code_block_formatting(self, content: str) -> str:
        """
        Fix code block formatting with precise structural control.
        Code blocks must be impeccable! 💎
        """
        # Fix triple backtick code blocks that should use double colon format
        lines = content.split('\n')
        in_code_block = False
        result = []
        
        for i, line in enumerate(lines):
            # Check for Python triple backtick blocks with indentation issues
            if line.strip().startswith('```python'):
                in_code_block = True
                # Replace with proper RST code block format - absolute precision
                indent = line[:len(line) - len(line.lstrip())]
                result.append(f"{indent}.. code-block:: python")
                result.append("")  # Empty line required
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                result.append("")  # Add an extra blank line after code blocks
            elif in_code_block:
                # Ensure consistent indentation for code inside blocks
                if line.strip():  # Only process non-empty lines
                    # Make sure code in blocks is indented by at least 4 spaces
                    indent = len(line) - len(line.lstrip())
                    if not indent:
                        result.append(f"    {line}")
                    else:
                        result.append(line)
                else:
                    result.append(line)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def add_no_index_to_duplicates(self, content: str) -> str:
        """
        Add :noindex: directive to duplicate object descriptions.
        Duplication is the enemy of clarity! ⚔️
        """
        # Find common duplicate patterns - usual suspects of duplication
        patterns = [
            r'(\.\. py:[a-z]+:: .*?base_url\n)',
            r'(\.\. py:[a-z]+:: .*?timeout\n)',
            r'(\.\. py:[a-z]+:: .*?max_retries\n)',
            r'(\.\. py:[a-z]+:: .*?__init__\(.*?\)\n)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            if len(matches) > 1:
                # Add :noindex: to all occurrences except the first one
                # First instance is canonical, others are mere echoes
                for match in matches[1:]:
                    directive = match.group(1)
                    # Only add :noindex: if it doesn't already have it
                    if ':noindex:' not in directive:
                        replacement = f"{directive}   :noindex:\n"
                        content = content[:match.start(1)] + replacement + content[match.end(1):]
        
        return content

if __name__ == "__main__":
    # Automatic detection - find your workspace
    script_dir = Path(__file__).resolve().parent
    docs_dir = script_dir
    
    # Accept override from command line
    if len(sys.argv) > 1:
        docs_dir = Path(sys.argv[1])
        
    print(f"🔍 Docstring Fixer initializing for {docs_dir}")
    fixer = DocstringFixer(docs_dir)
    fixed_count = fixer.fix_all_files()
    print(f"✅ Fixed formatting issues in {fixed_count} files")