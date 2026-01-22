import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_end_iferror(self):
    self.in_error = None
    self.skip_error = False
    self.skip_next = True