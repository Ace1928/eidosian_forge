from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def enable_commitfromnews(self):
    stack = config.GlobalStack()
    stack.set('commit.template_from_files', ['NEWS'])