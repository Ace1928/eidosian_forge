import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_def_finish(self, node, buffered, filtered, cached, callstack=True):
    """write the end section of a rendering function, either outermost or
        inline.

        this takes into account if the rendering function was filtered,
        buffered, etc.  and closes the corresponding try: block if any, and
        writes code to retrieve captured content, apply filters, send proper
        return value."""
    if not buffered and (not cached) and (not filtered):
        self.printer.writeline("return ''")
        if callstack:
            self.printer.writelines('finally:', 'context.caller_stack._pop_frame()', None)
    if buffered or filtered or cached:
        if buffered or cached:
            self.printer.writelines('finally:', '__M_buf = context._pop_buffer()')
        else:
            self.printer.writelines('finally:', '__M_buf, __M_writer = context._pop_buffer_and_writer()')
        if callstack:
            self.printer.writeline('context.caller_stack._pop_frame()')
        s = '__M_buf.getvalue()'
        if filtered:
            s = self.create_filter_callable(node.filter_args.args, s, False)
        self.printer.writeline(None)
        if buffered and (not cached):
            s = self.create_filter_callable(self.compiler.buffer_filters, s, False)
        if buffered or cached:
            self.printer.writeline('return %s' % s)
        else:
            self.printer.writelines('__M_writer(%s)' % s, "return ''")