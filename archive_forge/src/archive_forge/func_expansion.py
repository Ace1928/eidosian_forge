import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def expansion(self, *x):
    x, js = (x[:-1], x[-1])
    if js.children:
        js_code, = js.children
        js_code = js_code[2:-2]
        alias = '-> ' + self._new_function(js_code)
    else:
        alias = ''
    return ' '.join(x) + alias