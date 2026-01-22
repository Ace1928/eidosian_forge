import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def write_namespaces(self, namespaces):
    """write the module-level namespace-generating callable."""
    self.printer.writelines('def _mako_get_namespace(context, name):', 'try:', 'return context.namespaces[(__name__, name)]', 'except KeyError:', '_mako_generate_namespaces(context)', 'return context.namespaces[(__name__, name)]', None, None)
    self.printer.writeline('def _mako_generate_namespaces(context):')
    for node in namespaces.values():
        if 'import' in node.attributes:
            self.compiler.has_ns_imports = True
        self.printer.start_source(node.lineno)
        if len(node.nodes):
            self.printer.writeline('def make_namespace():')
            export = []
            identifiers = self.compiler.identifiers.branch(node)
            self.in_def = True

            class NSDefVisitor:

                def visitDefTag(s, node):
                    s.visitDefOrBase(node)

                def visitBlockTag(s, node):
                    s.visitDefOrBase(node)

                def visitDefOrBase(s, node):
                    if node.is_anonymous:
                        raise exceptions.CompileException("Can't put anonymous blocks inside <%namespace>", **node.exception_kwargs)
                    self.write_inline_def(node, identifiers, nested=False)
                    export.append(node.funcname)
            vis = NSDefVisitor()
            for n in node.nodes:
                n.accept_visitor(vis)
            self.printer.writeline('return [%s]' % ','.join(export))
            self.printer.writeline(None)
            self.in_def = False
            callable_name = 'make_namespace()'
        else:
            callable_name = 'None'
        if 'file' in node.parsed_attributes:
            self.printer.writeline('ns = runtime.TemplateNamespace(%r, context._clean_inheritance_tokens(), templateuri=%s, callables=%s,  calling_uri=_template_uri)' % (node.name, node.parsed_attributes.get('file', 'None'), callable_name))
        elif 'module' in node.parsed_attributes:
            self.printer.writeline('ns = runtime.ModuleNamespace(%r, context._clean_inheritance_tokens(), callables=%s, calling_uri=_template_uri, module=%s)' % (node.name, callable_name, node.parsed_attributes.get('module', 'None')))
        else:
            self.printer.writeline('ns = runtime.Namespace(%r, context._clean_inheritance_tokens(), callables=%s, calling_uri=_template_uri)' % (node.name, callable_name))
        if eval(node.attributes.get('inheritable', 'False')):
            self.printer.writeline("context['self'].%s = ns" % node.name)
        self.printer.writeline('context.namespaces[(__name__, %s)] = ns' % repr(node.name))
        self.printer.write_blanks(1)
    if not len(namespaces):
        self.printer.writeline('pass')
    self.printer.writeline(None)