import sys
import re
import copy
import time
import os.path
def expand_macros(self, tokens, expanded=None):
    if expanded is None:
        expanded = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == self.t_ID:
            if t.value in self.macros and t.value not in expanded:
                expanded[t.value] = True
                m = self.macros[t.value]
                if not m.arglist:
                    ex = self.expand_macros([copy.copy(_x) for _x in m.value], expanded)
                    for e in ex:
                        e.lineno = t.lineno
                    tokens[i:i + 1] = ex
                    i += len(ex)
                else:
                    j = i + 1
                    while j < len(tokens) and tokens[j].type in self.t_WS:
                        j += 1
                    if tokens[j].value == '(':
                        tokcount, args, positions = self.collect_args(tokens[j:])
                        if not m.variadic and len(args) != len(m.arglist):
                            self.error(self.source, t.lineno, 'Macro %s requires %d arguments' % (t.value, len(m.arglist)))
                            i = j + tokcount
                        elif m.variadic and len(args) < len(m.arglist) - 1:
                            if len(m.arglist) > 2:
                                self.error(self.source, t.lineno, 'Macro %s must have at least %d arguments' % (t.value, len(m.arglist) - 1))
                            else:
                                self.error(self.source, t.lineno, 'Macro %s must have at least %d argument' % (t.value, len(m.arglist) - 1))
                            i = j + tokcount
                        else:
                            if m.variadic:
                                if len(args) == len(m.arglist) - 1:
                                    args.append([])
                                else:
                                    args[len(m.arglist) - 1] = tokens[j + positions[len(m.arglist) - 1]:j + tokcount - 1]
                                    del args[len(m.arglist):]
                            rep = self.macro_expand_args(m, args)
                            rep = self.expand_macros(rep, expanded)
                            for r in rep:
                                r.lineno = t.lineno
                            tokens[i:j + tokcount] = rep
                            i += len(rep)
                del expanded[t.value]
                continue
            elif t.value == '__LINE__':
                t.type = self.t_INTEGER
                t.value = self.t_INTEGER_TYPE(t.lineno)
        i += 1
    return tokens