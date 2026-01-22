from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_mapping_key(self, first=False):
    if not self.check_token(FlowMappingEndToken):
        if not first:
            if self.check_token(FlowEntryToken):
                self.get_token()
            else:
                token = self.peek_token()
                raise ParserError('while parsing a flow mapping', self.marks[-1], "expected ',' or '}', but got %r" % token.id, token.start_mark)
        if self.check_token(KeyToken):
            token = self.get_token()
            if not self.check_token(ValueToken, FlowEntryToken, FlowMappingEndToken):
                self.states.append(self.parse_flow_mapping_value)
                return self.parse_flow_node()
            else:
                self.state = self.parse_flow_mapping_value
                return self.process_empty_scalar(token.end_mark)
        elif not self.check_token(FlowMappingEndToken):
            self.states.append(self.parse_flow_mapping_empty_value)
            return self.parse_flow_node()
    token = self.get_token()
    event = MappingEndEvent(token.start_mark, token.end_mark)
    self.state = self.states.pop()
    self.marks.pop()
    return event