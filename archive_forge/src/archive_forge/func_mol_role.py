import os
import traceback
import warnings
from os.path import join
from stat import ST_MTIME
import re
import runpy
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from subprocess import check_call, DEVNULL, CalledProcessError
from pathlib import Path
import matplotlib
def mol_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    n = []
    t = ''
    while text:
        if text[0] == '_':
            n.append(nodes.inline(text=t))
            t = ''
            m = re.match('\\d+', text[1:])
            if m is None:
                raise RuntimeError('Expected one or more digits after "_"')
            digits = m.group()
            n.append(nodes.subscript(text=digits))
            text = text[1 + len(digits):]
        else:
            t += text[0]
            text = text[1:]
    n.append(nodes.inline(text=t))
    return (n, [])