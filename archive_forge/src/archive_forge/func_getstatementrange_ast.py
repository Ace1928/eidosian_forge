import ast
from bisect import bisect_right
import inspect
import textwrap
import tokenize
import types
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union
import warnings
def getstatementrange_ast(lineno: int, source: Source, assertion: bool=False, astnode: Optional[ast.AST]=None) -> Tuple[ast.AST, int, int]:
    if astnode is None:
        content = str(source)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            astnode = ast.parse(content, 'source', 'exec')
    start, end = get_statement_startend2(lineno, astnode)
    if end is None:
        end = len(source.lines)
    if end > start + 1:
        block_finder = inspect.BlockFinder()
        block_finder.started = bool(source.lines[start]) and source.lines[start][0].isspace()
        it = (x + '\n' for x in source.lines[start:end])
        try:
            for tok in tokenize.generate_tokens(lambda: next(it)):
                block_finder.tokeneater(*tok)
        except (inspect.EndOfBlock, IndentationError):
            end = block_finder.last + start
        except Exception:
            pass
    while end:
        line = source.lines[end - 1].lstrip()
        if line.startswith('#') or not line:
            end -= 1
        else:
            break
    return (astnode, start, end)