import logging
def start_sphinx_py_method(self, method_name, parameters=None):
    self.new_paragraph()
    content = '.. py:method:: %s' % method_name
    if parameters is not None:
        content += '(%s)' % parameters
    self.doc.write(content)
    self.indent()
    self.new_paragraph()