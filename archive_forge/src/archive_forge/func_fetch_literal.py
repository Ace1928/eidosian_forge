from .error import MarkedYAMLError
from .tokens import *
def fetch_literal(self):
    self.fetch_block_scalar(style='|')