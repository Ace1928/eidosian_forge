import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
def process_filedata(self, module, data):
    stateMachineExpr = (stateMachine | namedStateMachine).ignore(pp.pythonStyleComment)
    generated_code = stateMachineExpr.transformString(data)
    if DEBUG:
        print(generated_code)
    COMPILE_MODE = 'exec'
    codeobj = compile(generated_code.rstrip(' \t'), module.__file__, COMPILE_MODE)
    exec(codeobj, module.__dict__)