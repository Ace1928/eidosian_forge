import sys
from argparse import ArgumentParser, FileType
from textwrap import indent
from logging import DEBUG, INFO, WARN, ERROR
from typing import Optional
import warnings
from lark import Lark, logger
def showwarning_as_comment(message, category, filename, lineno, file=None, line=None):
    text = warnings.formatwarning(message, category, filename, lineno, line)
    text = indent(text, '# ')
    if file is None:
        file = sys.stderr
        if file is None:
            return
    try:
        file.write(text)
    except OSError:
        pass