from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_block_node_or_indentless_sequence(self):
    return self.parse_node(block=True, indentless_sequence=True)