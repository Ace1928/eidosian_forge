import re
import ast
import logging
def extract_import_statements(self) -> list:
    """
        Extracts import statements using regex with detailed logging.

        Returns:
            list: A list of import statements extracted from the script content.
        """
    self.parser_logger.debug('Attempting to extract import statements.')
    import_statements = re.findall('^\\s*import .*', self.script_content, re.MULTILINE)
    self.parser_logger.info(f'Extracted {len(import_statements)} import statements.')
    return import_statements