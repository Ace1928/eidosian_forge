from codeop import CommandCompiler
from typing import Match
from itertools import tee, islice, chain
from ..lazyre import LazyReCompile
def leading_tabs_to_spaces(s: str) -> str:

    def tab_to_space(m: Match[str]) -> str:
        return len(m.group()) * 4 * ' '
    return '\n'.join((tabs_to_spaces_re.sub(tab_to_space, line) for line in s.split('\n')))