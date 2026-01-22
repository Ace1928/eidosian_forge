from codeop import CommandCompiler
from typing import Match
from itertools import tee, islice, chain
from ..lazyre import LazyReCompile
def indent_empty_lines(s: str, compiler: CommandCompiler) -> str:
    """Indents blank lines that would otherwise cause early compilation

    Only really works if starting on a new line"""
    initial_lines = s.split('\n')
    ends_with_newline = False
    if initial_lines and (not initial_lines[-1]):
        ends_with_newline = True
        initial_lines.pop()
    result_lines = []
    prevs, lines, nexts = tee(initial_lines, 3)
    prevs = chain(('',), prevs)
    nexts = chain(islice(nexts, 1, None), ('',))
    for p_line, line, n_line in zip(prevs, lines, nexts):
        if len(line) == 0:
            p_indent = indent_empty_lines_re.match(p_line).group()
            n_indent = indent_empty_lines_re.match(n_line).group()
            result_lines.append(min([p_indent, n_indent], key=len) + line)
        else:
            result_lines.append(line)
    return '\n'.join(result_lines) + ('\n' if ends_with_newline else '')