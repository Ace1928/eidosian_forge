import collections
import locale
import sys
from typing import Dict, List
from charset_normalizer import from_path  # pylint: disable=import-error
def do_detect_encoding(self, filename) -> None:
    """Return the detected file encoding.

        Parameters
        ----------
        filename : str
            The full path name of the file whose encoding is to be detected.
        """
    try:
        self.encoding = from_path(filename).best().encoding
        with self.do_open_with_encoding(filename) as check_file:
            check_file.read()
    except (SyntaxError, LookupError, UnicodeDecodeError):
        self.encoding = 'latin-1'