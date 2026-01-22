import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def parseopt_notrack(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
    lookahead = None
    lookaheadstack = []
    actions = self.action
    goto = self.goto
    prod = self.productions
    defaulted_states = self.defaulted_states
    pslice = YaccProduction(None)
    errorcount = 0
    if not lexer:
        from . import lex
        lexer = lex.lexer
    pslice.lexer = lexer
    pslice.parser = self
    if input is not None:
        lexer.input(input)
    if tokenfunc is None:
        get_token = lexer.token
    else:
        get_token = tokenfunc
    self.token = get_token
    statestack = []
    self.statestack = statestack
    symstack = []
    self.symstack = symstack
    pslice.stack = symstack
    errtoken = None
    statestack.append(0)
    sym = YaccSymbol()
    sym.type = '$end'
    symstack.append(sym)
    state = 0
    while True:
        if state not in defaulted_states:
            if not lookahead:
                if not lookaheadstack:
                    lookahead = get_token()
                else:
                    lookahead = lookaheadstack.pop()
                if not lookahead:
                    lookahead = YaccSymbol()
                    lookahead.type = '$end'
            ltype = lookahead.type
            t = actions[state].get(ltype)
        else:
            t = defaulted_states[state]
        if t is not None:
            if t > 0:
                statestack.append(t)
                state = t
                symstack.append(lookahead)
                lookahead = None
                if errorcount:
                    errorcount -= 1
                continue
            if t < 0:
                p = prod[-t]
                pname = p.name
                plen = p.len
                sym = YaccSymbol()
                sym.type = pname
                sym.value = None
                if plen:
                    targ = symstack[-plen - 1:]
                    targ[0] = sym
                    pslice.slice = targ
                    try:
                        del symstack[-plen:]
                        self.state = state
                        p.callable(pslice)
                        del statestack[-plen:]
                        symstack.append(sym)
                        state = goto[statestack[-1]][pname]
                        statestack.append(state)
                    except SyntaxError:
                        lookaheadstack.append(lookahead)
                        symstack.extend(targ[1:-1])
                        statestack.pop()
                        state = statestack[-1]
                        sym.type = 'error'
                        sym.value = 'error'
                        lookahead = sym
                        errorcount = error_count
                        self.errorok = False
                    continue
                else:
                    targ = [sym]
                    pslice.slice = targ
                    try:
                        self.state = state
                        p.callable(pslice)
                        symstack.append(sym)
                        state = goto[statestack[-1]][pname]
                        statestack.append(state)
                    except SyntaxError:
                        lookaheadstack.append(lookahead)
                        statestack.pop()
                        state = statestack[-1]
                        sym.type = 'error'
                        sym.value = 'error'
                        lookahead = sym
                        errorcount = error_count
                        self.errorok = False
                    continue
            if t == 0:
                n = symstack[-1]
                result = getattr(n, 'value', None)
                return result
        if t is None:
            if errorcount == 0 or self.errorok:
                errorcount = error_count
                self.errorok = False
                errtoken = lookahead
                if errtoken.type == '$end':
                    errtoken = None
                if self.errorfunc:
                    if errtoken and (not hasattr(errtoken, 'lexer')):
                        errtoken.lexer = lexer
                    self.state = state
                    tok = call_errorfunc(self.errorfunc, errtoken, self)
                    if self.errorok:
                        lookahead = tok
                        errtoken = None
                        continue
                elif errtoken:
                    if hasattr(errtoken, 'lineno'):
                        lineno = lookahead.lineno
                    else:
                        lineno = 0
                    if lineno:
                        sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                    else:
                        sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                else:
                    sys.stderr.write('yacc: Parse error in input. EOF\n')
                    return
            else:
                errorcount = error_count
            if len(statestack) <= 1 and lookahead.type != '$end':
                lookahead = None
                errtoken = None
                state = 0
                del lookaheadstack[:]
                continue
            if lookahead.type == '$end':
                return
            if lookahead.type != 'error':
                sym = symstack[-1]
                if sym.type == 'error':
                    lookahead = None
                    continue
                t = YaccSymbol()
                t.type = 'error'
                if hasattr(lookahead, 'lineno'):
                    t.lineno = t.endlineno = lookahead.lineno
                if hasattr(lookahead, 'lexpos'):
                    t.lexpos = t.endlexpos = lookahead.lexpos
                t.value = lookahead
                lookaheadstack.append(lookahead)
                lookahead = t
            else:
                sym = symstack.pop()
                statestack.pop()
                state = statestack[-1]
            continue
        raise RuntimeError('yacc: internal parser error!!!\n')