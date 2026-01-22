from .error import MarkedYAMLError
from .tokens import *
def fetch_block_entry(self):
    if not self.flow_level:
        if not self.allow_simple_key:
            raise ScannerError(None, None, 'sequence entries are not allowed here', self.get_mark())
        if self.add_indent(self.column):
            mark = self.get_mark()
            self.tokens.append(BlockSequenceStartToken(mark, mark))
    else:
        pass
    self.allow_simple_key = True
    self.remove_possible_simple_key()
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(BlockEntryToken(start_mark, end_mark))