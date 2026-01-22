import sys
import re
import copy
import time
import os.path
def evalexpr(self, tokens):
    i = 0
    while i < len(tokens):
        if tokens[i].type == self.t_ID and tokens[i].value == 'defined':
            j = i + 1
            needparen = False
            result = '0L'
            while j < len(tokens):
                if tokens[j].type in self.t_WS:
                    j += 1
                    continue
                elif tokens[j].type == self.t_ID:
                    if tokens[j].value in self.macros:
                        result = '1L'
                    else:
                        result = '0L'
                    if not needparen:
                        break
                elif tokens[j].value == '(':
                    needparen = True
                elif tokens[j].value == ')':
                    break
                else:
                    self.error(self.source, tokens[i].lineno, 'Malformed defined()')
                j += 1
            tokens[i].type = self.t_INTEGER
            tokens[i].value = self.t_INTEGER_TYPE(result)
            del tokens[i + 1:j + 1]
        i += 1
    tokens = self.expand_macros(tokens)
    for i, t in enumerate(tokens):
        if t.type == self.t_ID:
            tokens[i] = copy.copy(t)
            tokens[i].type = self.t_INTEGER
            tokens[i].value = self.t_INTEGER_TYPE('0L')
        elif t.type == self.t_INTEGER:
            tokens[i] = copy.copy(t)
            tokens[i].value = str(tokens[i].value)
            while tokens[i].value[-1] not in '0123456789abcdefABCDEF':
                tokens[i].value = tokens[i].value[:-1]
    expr = ''.join([str(x.value) for x in tokens])
    expr = expr.replace('&&', ' and ')
    expr = expr.replace('||', ' or ')
    expr = expr.replace('!', ' not ')
    try:
        result = eval(expr)
    except Exception:
        self.error(self.source, tokens[0].lineno, "Couldn't evaluate expression")
        result = 0
    return result