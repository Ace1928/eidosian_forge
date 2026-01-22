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
def git_role_tmpl(urlroot, role, rawtext, text, lineno, inliner, options={}, content=[]):
    if text[-1] == '>':
        i = text.index('<')
        name = text[:i - 1]
        text = text[i + 1:-1]
    else:
        name = text
        if name[0] == '~':
            name = name.split('/')[-1]
            text = text[1:]
        if '?' in name:
            name = name[:name.index('?')]
    is_tag = text.startswith('..')
    path = os.path.join('..', text)
    do_exists = os.path.exists(path)
    if not (is_tag or do_exists):
        msg = 'Broken link: {}: Non-existing path: {}'.format(rawtext, path)
        msg = inliner.reporter.error(msg, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    ref = urlroot + text
    set_classes(options)
    node = nodes.reference(rawtext, name, refuri=ref, **options)
    return ([node], [])