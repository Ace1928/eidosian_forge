from pyparsing import *
from sys import stdin, argv, exit
def newline_label(self, name, internal=False, definition=False):
    """Inserts a newline and a label (helper function)
           name - label name
           internal - boolean value, adds "@" prefix to label
           definition - boolean value, adds ":" suffix to label
        """
    self.newline_text(self.label('{0}{1}{2}'.format('@' if internal else '', name, ':' if definition else '')))