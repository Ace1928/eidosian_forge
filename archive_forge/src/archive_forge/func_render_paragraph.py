import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def render_paragraph(paragraph, meta_variables):
    if paragraph.startswith(USAGE_MARKER):
        tokens = paragraph.split()
        return '**%s** `%s`' % (tokens[0], ' '.join(tokens[1:]))
    if paragraph == 'Supported options:':
        return '**%s**' % paragraph
    if re.match('^\\s*\\$\\s+\\S', paragraph):
        lines = paragraph.splitlines()
        if not paragraph[0].isspace():
            lines = ['  %s' % line for line in lines]
        lines.insert(0, '.. code-block:: sh')
        lines.insert(1, '')
        return '\n'.join(lines)
    if not paragraph[0].isspace():
        paragraph = re.sub("`(.+?)'", '"\\1"', paragraph)
        paragraph = paragraph.replace('*', '\\*')
        paragraph = replace_special_tokens(paragraph, meta_variables, lambda token: '``%s``' % token)
    return paragraph