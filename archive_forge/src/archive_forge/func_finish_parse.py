import sys
from docutils import Component
def finish_parse(self):
    """Finalize parse details.  Call at end of `self.parse()`."""
    self.document.reporter.detach_observer(self.document.note_parse_message)