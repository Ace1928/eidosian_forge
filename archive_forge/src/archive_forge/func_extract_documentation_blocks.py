import re
import ast
import logging
def extract_documentation_blocks(self) -> list:
    """
        Extracts block and inline documentation with detailed logging.

        Returns:
            list: A list of documentation blocks and inline comments extracted from the script content.
        """
    self.parser_logger.debug('Attempting to extract documentation blocks and inline comments.')
    documentation_blocks = re.findall('""".*?"""|\\\'\\\'\\\'.*?\\\'\\\'\\\'|#.*$', self.script_content, re.MULTILINE | re.DOTALL)
    self.parser_logger.info(f'Extracted {len(documentation_blocks)} documentation blocks.')
    return documentation_blocks