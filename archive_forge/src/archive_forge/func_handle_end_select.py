import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_end_select(self):
    self.write_text('</select>')
    self.skip_next = True
    if not self.prefix_error and self.in_select:
        self.write_marker(self.in_select)
    self.in_select = None