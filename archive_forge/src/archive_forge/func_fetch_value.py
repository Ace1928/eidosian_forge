from .error import MarkedYAMLError
from .tokens import *
def fetch_value(self):
    if self.flow_level in self.possible_simple_keys:
        key = self.possible_simple_keys[self.flow_level]
        del self.possible_simple_keys[self.flow_level]
        self.tokens.insert(key.token_number - self.tokens_taken, KeyToken(key.mark, key.mark))
        if not self.flow_level:
            if self.add_indent(key.column):
                self.tokens.insert(key.token_number - self.tokens_taken, BlockMappingStartToken(key.mark, key.mark))
        self.allow_simple_key = False
    else:
        if not self.flow_level:
            if not self.allow_simple_key:
                raise ScannerError(None, None, 'mapping values are not allowed here', self.get_mark())
        if not self.flow_level:
            if self.add_indent(self.column):
                mark = self.get_mark()
                self.tokens.append(BlockMappingStartToken(mark, mark))
        self.allow_simple_key = not self.flow_level
        self.remove_possible_simple_key()
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(ValueToken(start_mark, end_mark))