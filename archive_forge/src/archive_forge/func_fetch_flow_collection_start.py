from .error import MarkedYAMLError
from .tokens import *
def fetch_flow_collection_start(self, TokenClass):
    self.save_possible_simple_key()
    self.flow_level += 1
    self.allow_simple_key = True
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(TokenClass(start_mark, end_mark))