from .error import MarkedYAMLError
from .tokens import *
def save_possible_simple_key(self):
    required = not self.flow_level and self.indent == self.column
    if self.allow_simple_key:
        self.remove_possible_simple_key()
        token_number = self.tokens_taken + len(self.tokens)
        key = SimpleKey(token_number, required, self.index, self.line, self.column, self.get_mark())
        self.possible_simple_keys[self.flow_level] = key