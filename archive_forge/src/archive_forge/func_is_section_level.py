import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def is_section_level(self, level, section):
    return self._level_to_elem.get(level, None) == section