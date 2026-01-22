import re
import sys
import warnings
from typing import Dict, List, Tuple
from docutils.parsers.rst.states import Body
from sphinx.deprecation import RemovedInSphinx60Warning
def prepare_commentdoc(s: str) -> List[str]:
    """Extract documentation comment lines (starting with #:) and return them
    as a list of lines.  Returns an empty list if there is no documentation.
    """
    result = []
    lines = [line.strip() for line in s.expandtabs().splitlines()]
    for line in lines:
        if line.startswith('#:'):
            line = line[2:]
            if line and line[0] == ' ':
                line = line[1:]
            result.append(line)
    if result and result[-1]:
        result.append('')
    return result