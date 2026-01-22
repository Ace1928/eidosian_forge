from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_block_node(self):
    return self.parse_node(block=True)