from typing import Optional
from docutils import nodes
from sphinx.builders.html import HTMLTranslator
def wrap_displaymath(text: str, label: Optional[str], numbering: bool) -> str:

    def is_equation(part: str) -> str:
        return part.strip()
    if label is None:
        labeldef = ''
    else:
        labeldef = '\\label{%s}' % label
        numbering = True
    parts = list(filter(is_equation, text.split('\n\n')))
    equations = []
    if len(parts) == 0:
        return ''
    elif len(parts) == 1:
        if numbering:
            begin = '\\begin{equation}' + labeldef
            end = '\\end{equation}'
        else:
            begin = '\\begin{equation*}' + labeldef
            end = '\\end{equation*}'
        equations.append('\\begin{split}%s\\end{split}\n' % parts[0])
    else:
        if numbering:
            begin = '\\begin{align}%s\\!\\begin{aligned}' % labeldef
            end = '\\end{aligned}\\end{align}'
        else:
            begin = '\\begin{align*}%s\\!\\begin{aligned}' % labeldef
            end = '\\end{aligned}\\end{align*}'
        for part in parts:
            equations.append('%s\\\\\n' % part.strip())
    return '%s\n%s%s' % (begin, ''.join(equations), end)