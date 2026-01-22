from typing import Optional, Tuple
from cirq import ops, protocols
def escape_text_for_latex(text):
    escaped = text.replace('\\', '\\textbackslash{}').replace('{', '\\{').replace('}', '\\}').replace('^', '\\textasciicircum{}').replace('~', '\\textasciitilde{}').replace('_', '\\_').replace('$', '\\$').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')
    return '\\text{' + escaped + '}'