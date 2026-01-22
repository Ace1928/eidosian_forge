from .error import MarkedYAMLError
from .tokens import *
def need_more_tokens(self):
    if self.done:
        return False
    if not self.tokens:
        return True
    self.stale_possible_simple_keys()
    if self.next_possible_simple_key() == self.tokens_taken:
        return True