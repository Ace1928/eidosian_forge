import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def do_format_standard_in(self, parser: argparse.ArgumentParser):
    """Print formatted text to standard out.

        Parameters
        ----------
        parser: argparse.ArgumentParser
            The argument parser containing the formatting options.
        """
    if len(self.args.files) > 1:
        parser.error('cannot mix standard in and regular files')
    if self.args.in_place:
        parser.error('--in-place cannot be used with standard input')
    if self.args.recursive:
        parser.error('--recursive cannot be used with standard input')
    encoding = None
    source = self.stdin.read()
    if not isinstance(source, unicode):
        encoding = self.stdin.encoding or self.encodor.system_encoding
        source = source.decode(encoding)
    formatted_source = self._do_format_code(source)
    if encoding:
        formatted_source = formatted_source.encode(encoding)
    self.stdout.write(formatted_source)