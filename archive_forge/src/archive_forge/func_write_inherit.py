import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_inherit(self, node):
    """write the module-level inheritance-determination callable."""
    self.printer.writelines('def _mako_inherit(template, context):', '_mako_generate_namespaces(context)', 'return runtime._inherit_from(context, %s, _template_uri)' % node.parsed_attributes['file'], None)