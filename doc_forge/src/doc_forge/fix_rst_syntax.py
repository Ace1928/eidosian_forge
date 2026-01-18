#!/usr/bin/env python3
# 🌀 Eidosian RST Syntax Perfector
"""
RST Syntax Perfector - Fixing Documentation Markup with Surgical Precision

This script repairs common RST syntax issues in documentation files:
1. Unmatched inline literals (`text without closing backtick)
2. Malformed directive patterns and code blocks
3. Missing or improper indentation in nested structures
4. Backtick balance issues that cause Sphinx heartburn

Following Eidosian principles of:
- Precision as Style: Each markup element must be perfectly formed
- Contextual Integrity: Formatting fixes respect document meaning and structure
- Flow Like a River: Changes preserve the natural documentation narrative
"""

import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Counter
from functools import lru_cache

# 📊 Self-aware logging - know thyself!
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("eidosian_docs.rst_syntax_perfector")

class RstSyntaxPerfector:
    """
    Performs surgical corrections on RST syntax with architectural precision.
    Like a typographer who balances every element for perfect harmony! 🎭
    """
    
    def __init__(self, docs_dir: Path = Path(".")):
        """Initialize the syntax perfector with a documentation directory."""
        self.docs_dir = Path(docs_dir)
        self.fixed_count = 0
        self.issue_tracker = Counter()  # 📊 Track issues by type for reporting
        
        # Regex patterns tuned for precision and performance
        self.patterns = {
            "unclosed_backtick": re.compile(r'(`[^`\n]+)(?=\n|\s\s|\))'),
            "unmatched_backticks": re.compile(r'^(.*?)`([^`]+)$'),
            "triple_backtick": re.compile(r'```(?:python)?(.*?)```', re.DOTALL),
            "indentation_issue": re.compile(r'(\n\s+\S.*\n)(\s{1,3}\S)'),
            "double_backtick_code": re.compile(r'``([^`]+)``'),
            "malformed_directive": re.compile(r'(\.\.\s+[a-z]+::.*?)(?:\n)(?:\S)'),
        }
    
    def fix_inline_text_issues(self, target_path: Optional[Path] = None) -> int:
        """
        Fix inline interpreted text syntax issues in RST files.
        
        Args:
            target_path: Optional specific file to fix, otherwise scans docs_dir
            
        Returns:
            Number of files fixed (victory count! 🏆)
        """
        self.fixed_count = 0
        
        # Determine what to process - specific target or full directory scan
        if target_path and target_path.exists():
            files_to_process = [target_path]
        else:
            # Default: Focus on autoapi and exceptions documentation
            exceptions_paths = list(self.docs_dir.glob("**/exceptions/**/*.rst"))
            autoapi_paths = list(self.docs_dir.glob("**/autoapi/**/*.rst"))
            api_paths = list(self.docs_dir.glob("**/api/**/*.rst"))
            files_to_process = exceptions_paths + autoapi_paths + api_paths
            
            if not files_to_process:
                logger.warning(f"⚠️ No RST files found to process in {self.docs_dir}")
                return 0
        
        # Process each file - surgical precision for each document
        for rst_file in files_to_process:
            try:
                if self._fix_file(rst_file):
                    self.fixed_count += 1
            except Exception as e:
                logger.error(f"❌ Error processing {rst_file}: {e}")
        
        # Report results - celebrate victories!
        if self.fixed_count > 0:
            logger.info(f"✅ Fixed RST syntax issues in {self.fixed_count} files")
            # Report specific issue types fixed
            for issue_type, count in self.issue_tracker.most_common():
                logger.info(f"  • {issue_type}: {count} issues fixed")
        else:
            logger.info("✨ No RST syntax issues found - your docs are impeccable!")
            
        return self.fixed_count
    
    @lru_cache(maxsize=32)  # 🚀 Cache regex results for velocity
    def _get_syntax_pattern(self, pattern_name: str):
        """Get cached regex pattern for performance."""
        return self.patterns.get(pattern_name)
    
    def _fix_file(self, file_path: Path) -> bool:
        """
        Fix all syntax issues in a single file with artistic precision.
        
        Args:
            file_path: Path to the RST file to process
            
        Returns:
            True if file was modified, False otherwise
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Store original for comparison - don't fix what isn't broken
            original_content = content
            
            # Apply fixes with cascading precision - each building on the last
            content = self._fix_unclosed_backticks(content)
            content = self._fix_triple_backticks(content)
            content = self._fix_indentation(content)
            content = self._fix_directives(content)
            content = self._fix_base_inheritance(content)
            content = self._fix_code_blocks(content)
            
            # Only write back if changes were made - efficient I/O
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.debug(f"📝 Fixed syntax issues in {file_path.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error fixing {file_path}: {e}")
            return False
    
    def _fix_unclosed_backticks(self, content: str) -> str:
        """
        Fix unclosed backticks in inline interpreted text.
        The asymmetry of unclosed backticks is like fingernails on a chalkboard! 😖
        """
        # Fix pattern: `some text without closing backtick
        pattern = self._get_syntax_pattern("unclosed_backtick")
        matches = list(pattern.finditer(content))
        
        # Apply fixes in reverse order to avoid position shifts
        for match in reversed(matches):
            backtick_text = match.group(1)
            if not backtick_text.endswith('`'):
                replacement = f"{backtick_text}`"
                content = content[:match.start(1)] + replacement + content[match.end(1):]
                self.issue_tracker["unclosed_backticks"] += 1
        
        # Fix backtick spans across lines (line-by-line approach)
        lines = content.split('\n')
        for i in range(len(lines) - 1):
            # Count backticks to find odd numbers (unbalanced)
            if lines[i].count('`') % 2 == 1 and '`' in lines[i]:
                # Check if next line also has odd backticks - they might be related
                if i + 1 < len(lines) and lines[i+1].count('`') % 2 == 1:
                    # If line ends with backtick, it might be properly closed already
                    if not lines[i].rstrip().endswith('`'):
                        lines[i] += '`'
                        self.issue_tracker["line_spanning_backticks"] += 1
                # Otherwise add closing backtick to this line
                else:
                    if not lines[i].rstrip().endswith('`'):
                        lines[i] += '`'
                        self.issue_tracker["lone_unclosed_backticks"] += 1
        
        return '\n'.join(lines)
    
    def _fix_triple_backticks(self, content: str) -> str:
        """
        Convert triple backtick code blocks to proper RST syntax.
        Markdown is charming, but this is RST territory! 🏰
        """
        pattern = self._get_syntax_pattern("triple_backtick")
        
        # Function to process each match and convert to RST format
        def replace_triple_backticks(match):
            code_content = match.group(1).strip()
            self.issue_tracker["triple_backtick_blocks"] += 1
            
            # Proper RST code block with correct indentation
            rst_block = "\n.. code-block:: python\n\n"
            for line in code_content.split('\n'):
                rst_block += f"    {line}\n"
            return rst_block
        
        # Apply replacements
        return pattern.sub(replace_triple_backticks, content)
    
    def _fix_indentation(self, content: str) -> str:
        """
        Fix inconsistent indentation in code blocks and directives.
        Indentation is the backbone of structure - keep it straight! 📏
        """
        # First, fix general indentation issues
        pattern = self._get_syntax_pattern("indentation_issue")
        
        def fix_indent(match):
            self.issue_tracker["indentation"] += 1
            return f"{match.group(1)}    {match.group(2).lstrip()}"
        
        content = pattern.sub(fix_indent, content)
        
        # Then fix specific code block indentation
        lines = content.split('\n')
        in_code_block = False
        result = []
        
        for i, line in enumerate(lines):
            # Detect code block start
            if line.strip() == ".. code-block::" or line.rstrip().endswith("::"):
                in_code_block = True
                result.append(line)
                # Ensure blank line after directive
                if i+1 < len(lines) and lines[i+1].strip():
                    result.append("")
                    self.issue_tracker["code_block_spacing"] += 1
                continue
                
            # Fix indentation in code blocks
            if in_code_block and line.strip():
                # Check for proper indentation (4 spaces)
                indent = len(line) - len(line.lstrip())
                if 0 < indent < 4:
                    # Fix insufficient indentation
                    result.append("    " + line.lstrip())
                    self.issue_tracker["code_block_indentation"] += 1
                    continue
                elif indent == 0 and not line.startswith(".."):
                    # Likely no longer in code block
                    in_code_block = False
            
            # End of code block detection
            if in_code_block and not line.strip():
                result.append(line)
                if i+1 < len(lines) and not lines[i+1].startswith((' ', '\t')):
                    in_code_block = False
                continue
                
            result.append(line)
            
        return '\n'.join(result)
    
    def _fix_directives(self, content: str) -> str:
        """
        Fix malformed directives that don't have proper spacing.
        Directives need breathing room to work their magic! 💨
        """
        pattern = self._get_syntax_pattern("malformed_directive")
        
        def fix_directive(match):
            self.issue_tracker["directive_spacing"] += 1
            return f"{match.group(1)}\n\n"
        
        return pattern.sub(fix_directive, content)
    
    def _fix_base_inheritance(self, content: str) -> str:
        """
        Fix specific issues with base inheritance references.
        Inheritance is the family tree of code - keep it pristine! 🌳
        """
        # Fix 'Bases: :py:obj:exception.Exception`' pattern
        base_pattern = r'(Bases: :py:obj:`[^`]+)(?!\`)'
        
        def fix_base_reference(match):
            self.issue_tracker["inheritance_references"] += 1
            return f"{match.group(1)}`"
        
        return re.sub(base_pattern, fix_base_reference, content)
    
    def _fix_code_blocks(self, content: str) -> str:
        """
        Fix issues with code block formatting and double backticks.
        Code blocks should be fortresses of clarity! 🏯
        """
        # Fix double backtick code that should be single
        double_pattern = self._get_syntax_pattern("double_backtick_code")
        
        def fix_double_backticks(match):
            code = match.group(1)
            # Only fix if it doesn't contain any backticks already
            if '`' not in code:
                self.issue_tracker["double_backticks"] += 1
                return f"`{code}`"
            return match.group(0)
        
        content = double_pattern.sub(fix_double_backticks, content)
        
        # Ensure code block directives have proper double colons
        content = re.sub(r'(\.\.)\s+code-block\s*:([^:])', r'\1 code-block::\2', content)
        
        return content

def main():
    """Main entry point with command-line interface magic."""
    # Display our beautiful ASCII art banner - because docs deserve style!
    print("""
╭──────────────────────────────────────────────────╮
│     🌀 EIDOSIAN RST SYNTAX PERFECTOR v2.1 🌀     │
│   Making documentation beautiful one backtick     │
│              at a time since 2025                │
╰──────────────────────────────────────────────────╯
    """)
    
    # Parse command-line arguments
    docs_dir = Path(".")
    target_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("\n🌟 Usage Options:")
            print("  fix_rst_syntax.py [docs_dir]")
            print("  fix_rst_syntax.py --file <specific_file.rst>")
            print("\nExamples:")
            print("  fix_rst_syntax.py /path/to/docs")
            print("  fix_rst_syntax.py --file /path/to/docs/api/issues.rst")
            sys.exit(0)
        elif sys.argv[1] == "--file" and len(sys.argv) > 2:
            target_file = Path(sys.argv[2])
            docs_dir = target_file.parent
        else:
            docs_dir = Path(sys.argv[1])
    
    # Create and run our syntax perfector!
    fixer = RstSyntaxPerfector(docs_dir)
    fixed = fixer.fix_inline_text_issues(target_file)
    
    # Final report - because metrics matter!
    if fixed > 0:
        print(f"\n🎉 Success! Fixed RST syntax issues in {fixed} files")
    else:
        print("\n✨ No RST syntax issues found - your docs are already perfect!")
    
    # Add a helpful tip
    print("\n💡 TIP: Run a Sphinx build to verify all markup renders correctly!")

if __name__ == "__main__":
    main()
