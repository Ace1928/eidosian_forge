from .error import MarkedYAMLError
from .tokens import *
def scan_flow_scalar(self, style):
    if style == '"':
        double = True
    else:
        double = False
    chunks = []
    start_mark = self.get_mark()
    quote = self.peek()
    self.forward()
    chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
    while self.peek() != quote:
        chunks.extend(self.scan_flow_scalar_spaces(double, start_mark))
        chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
    self.forward()
    end_mark = self.get_mark()
    return ScalarToken(''.join(chunks), False, start_mark, end_mark, style)