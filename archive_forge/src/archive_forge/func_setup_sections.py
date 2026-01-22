import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def setup_sections(self):
    self._level_to_elem = {0: self.document}