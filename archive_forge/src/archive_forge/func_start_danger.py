import logging
def start_danger(self, attrs=None):
    self.new_paragraph()
    self.doc.write('.. danger::')
    self.indent()
    self.new_paragraph()