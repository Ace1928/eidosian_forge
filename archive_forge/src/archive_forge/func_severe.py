import docutils.parsers
import docutils.statemachine
from docutils.parsers.rst import states
from docutils import frontend, nodes, Component
from docutils.transforms import universal
def severe(self, message):
    return self.directive_error(4, message)