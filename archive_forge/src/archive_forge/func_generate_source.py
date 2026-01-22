from string import Template
import sys
def generate_source(self):
    src = self._gen_init()
    src += '\n' + self._gen_children()
    src += '\n' + self._gen_iter()
    src += '\n' + self._gen_attr_names()
    return src