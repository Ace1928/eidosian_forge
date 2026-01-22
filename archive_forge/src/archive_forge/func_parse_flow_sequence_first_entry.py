from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_sequence_first_entry(self):
    token = self.get_token()
    self.marks.append(token.start_mark)
    return self.parse_flow_sequence_entry(first=True)