import sys
from docutils import Component
def setup_parse(self, inputstring, document):
    """Initial parse setup.  Call at start of `self.parse()`."""
    self.inputstring = inputstring
    self.document = document
    document.reporter.attach_observer(document.note_parse_message)