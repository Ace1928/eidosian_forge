import sys
from types import CodeType
import six
def build_code_chunk(code, filename, name, lineno):
    if hasattr(code, 'replace'):
        return code.replace(co_filename=filename, co_name=name, co_firstlineno=lineno)
    return CodeType(0, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags | 64, code.co_code, code.co_consts, code.co_names, code.co_varnames, filename, name, lineno, code.co_lnotab, (), ())