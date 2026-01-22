from .error import MarkedYAMLError
from .tokens import *
def fetch_flow_scalar(self, style):
    self.save_possible_simple_key()
    self.allow_simple_key = False
    self.tokens.append(self.scan_flow_scalar(style))