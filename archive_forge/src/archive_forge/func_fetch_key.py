from .error import MarkedYAMLError
from .tokens import *
def fetch_key(self):
    if not self.flow_level:
        if not self.allow_simple_key:
            raise ScannerError(None, None, 'mapping keys are not allowed here', self.get_mark())
        if self.add_indent(self.column):
            mark = self.get_mark()
            self.tokens.append(BlockMappingStartToken(mark, mark))
    self.allow_simple_key = not self.flow_level
    self.remove_possible_simple_key()
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(KeyToken(start_mark, end_mark))