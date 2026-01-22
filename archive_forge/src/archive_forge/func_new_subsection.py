import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def new_subsection(self, title, lineno, messages):
    """Append new subsection to document tree. On return, check level."""
    memo = self.memo
    mylevel = memo.section_level
    memo.section_level += 1
    section_node = nodes.section()
    self.parent += section_node
    textnodes, title_messages = self.inline_text(title, lineno)
    titlenode = nodes.title(title, '', *textnodes)
    name = normalize_name(titlenode.astext())
    section_node['names'].append(name)
    section_node += titlenode
    section_node += messages
    section_node += title_messages
    self.document.note_implicit_target(section_node, section_node)
    offset = self.state_machine.line_offset + 1
    absoffset = self.state_machine.abs_line_offset() + 1
    newabsoffset = self.nested_parse(self.state_machine.input_lines[offset:], input_offset=absoffset, node=section_node, match_titles=True)
    self.goto_line(newabsoffset)
    if memo.section_level <= mylevel:
        raise EOFError
    memo.section_level = mylevel