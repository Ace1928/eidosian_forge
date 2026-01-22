from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_mapping_empty_value(self):
    self.state = self.parse_flow_mapping_key
    return self.process_empty_scalar(self.peek_token().start_mark)