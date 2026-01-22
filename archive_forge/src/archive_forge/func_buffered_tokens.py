import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
def buffered_tokens(self):
    """
        Generator of unprocessed tokens after doing insertions and before
        changing to a new state.

        """
    if self.mode == 'output':
        tokens = [(0, Generic.Output, self.buffer)]
    elif self.mode == 'input':
        tokens = self.pylexer.get_tokens_unprocessed(self.buffer)
    else:
        tokens = self.tblexer.get_tokens_unprocessed(self.buffer)
    for i, t, v in do_insertions(self.insertions, tokens):
        yield (self.index + i, t, v)
    self.index += len(self.buffer)
    self.buffer = u''
    self.insertions = []