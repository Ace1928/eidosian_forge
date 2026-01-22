import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_end_textarea(self):
    if self.skip_textarea:
        self.skip_textarea = False
    else:
        self.write_text('</textarea>')
    self.in_textarea = False
    self.skip_next = True
    if not self.prefix_error:
        self.write_marker(self.last_textarea_name)
    self.last_textarea_name = None