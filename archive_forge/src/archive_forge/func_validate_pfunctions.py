import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def validate_pfunctions(self):
    grammar = []
    if len(self.pfuncs) == 0:
        self.log.error('no rules of the form p_rulename are defined')
        self.error = True
        return
    for line, module, name, doc in self.pfuncs:
        file = inspect.getsourcefile(module)
        func = self.pdict[name]
        if isinstance(func, types.MethodType):
            reqargs = 2
        else:
            reqargs = 1
        if func.__code__.co_argcount > reqargs:
            self.log.error('%s:%d: Rule %r has too many arguments', file, line, func.__name__)
            self.error = True
        elif func.__code__.co_argcount < reqargs:
            self.log.error('%s:%d: Rule %r requires an argument', file, line, func.__name__)
            self.error = True
        elif not func.__doc__:
            self.log.warning('%s:%d: No documentation string specified in function %r (ignored)', file, line, func.__name__)
        else:
            try:
                parsed_g = parse_grammar(doc, file, line)
                for g in parsed_g:
                    grammar.append((name, g))
            except SyntaxError as e:
                self.log.error(str(e))
                self.error = True
            self.modules.add(module)
    for n, v in self.pdict.items():
        if n.startswith('p_') and isinstance(v, (types.FunctionType, types.MethodType)):
            continue
        if n.startswith('t_'):
            continue
        if n.startswith('p_') and n != 'p_error':
            self.log.warning('%r not defined as a function', n)
        if isinstance(v, types.FunctionType) and v.__code__.co_argcount == 1 or (isinstance(v, types.MethodType) and v.__func__.__code__.co_argcount == 2):
            if v.__doc__:
                try:
                    doc = v.__doc__.split(' ')
                    if doc[1] == ':':
                        self.log.warning('%s:%d: Possible grammar rule %r defined without p_ prefix', v.__code__.co_filename, v.__code__.co_firstlineno, n)
                except IndexError:
                    pass
    self.grammar = grammar