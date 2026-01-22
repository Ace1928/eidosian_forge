from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_document_end(self):
    token = self.peek_token()
    start_mark = end_mark = token.start_mark
    explicit = False
    if self.check_token(DocumentEndToken):
        token = self.get_token()
        end_mark = token.end_mark
        explicit = True
    event = DocumentEndEvent(start_mark, end_mark, explicit=explicit)
    self.state = self.parse_document_start
    return event