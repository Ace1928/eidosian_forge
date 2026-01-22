import logging
def end_ul(self):
    self.list_depth -= 1
    if self.list_depth != 0:
        self.dedent()
    self.new_paragraph()