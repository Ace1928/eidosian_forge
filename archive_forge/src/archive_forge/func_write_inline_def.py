import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_inline_def(self, node, identifiers, nested):
    """write a locally-available def callable inside an enclosing def."""
    namedecls = node.get_argument_expressions()
    decorator = node.decorator
    if decorator:
        self.printer.writeline('@runtime._decorate_inline(context, %s)' % decorator)
    self.printer.writeline('def %s(%s):' % (node.funcname, ','.join(namedecls)))
    filtered = len(node.filter_args.args) > 0
    buffered = eval(node.attributes.get('buffered', 'False'))
    cached = eval(node.attributes.get('cached', 'False'))
    self.printer.writelines('__M_caller = context.caller_stack._push_frame()', 'try:')
    if buffered or filtered or cached:
        self.printer.writelines('context._push_buffer()')
    identifiers = identifiers.branch(node, nested=nested)
    self.write_variable_declares(identifiers)
    self.identifier_stack.append(identifiers)
    for n in node.nodes:
        n.accept_visitor(self)
    self.identifier_stack.pop()
    self.write_def_finish(node, buffered, filtered, cached)
    self.printer.writeline(None)
    if cached:
        self.write_cache_decorator(node, node.funcname, namedecls, False, identifiers, inline=True, toplevel=False)