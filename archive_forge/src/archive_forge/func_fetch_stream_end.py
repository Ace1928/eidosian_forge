from .error import MarkedYAMLError
from .tokens import *
def fetch_stream_end(self):
    self.unwind_indent(-1)
    self.remove_possible_simple_key()
    self.allow_simple_key = False
    self.possible_simple_keys = {}
    mark = self.get_mark()
    self.tokens.append(StreamEndToken(mark, mark))
    self.done = True