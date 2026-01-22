import re
import sys
import warnings
from typing import Dict, List, Tuple
from docutils.parsers.rst.states import Body
from sphinx.deprecation import RemovedInSphinx60Warning
def separate_metadata(s: str) -> Tuple[str, Dict[str, str]]:
    """Separate docstring into metadata and others."""
    in_other_element = False
    metadata: Dict[str, str] = {}
    lines = []
    if not s:
        return (s, metadata)
    for line in prepare_docstring(s):
        if line.strip() == '':
            in_other_element = False
            lines.append(line)
        else:
            matched = field_list_item_re.match(line)
            if matched and (not in_other_element):
                field_name = matched.group()[1:].split(':', 1)[0]
                if field_name.startswith('meta '):
                    name = field_name[5:].strip()
                    metadata[name] = line[matched.end():].strip()
                else:
                    lines.append(line)
            else:
                in_other_element = True
                lines.append(line)
    return ('\n'.join(lines), metadata)