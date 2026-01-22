import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def depart_heading(self, _):
    """Finish establishing section

        Wrap up title node, but stick in the section node. Add the section names
        based on all the text nodes added to the title.
        """
    assert isinstance(self.current_node, nodes.title)
    text = self.current_node.astext()
    if self.translate_section_name:
        text = self.translate_section_name(text)
    name = nodes.fully_normalize_name(text)
    section = self.current_node.parent
    section['names'].append(name)
    self.document.note_implicit_target(section, section)
    self.current_node = section