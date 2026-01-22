import re
from pygments.lexer import RegexLexer, bygroups, default, include, using, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, \
from pygments.lexers._csound_builtins import OPCODES
from pygments.lexers.html import HtmlLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.scripting import LuaLexer
def name_callback(lexer, match):
    name = match.group(0)
    if re.match('p\\d+$', name) or name in OPCODES:
        yield (match.start(), Name.Builtin, name)
    elif name in lexer.user_defined_opcodes:
        yield (match.start(), Name.Function, name)
    else:
        nameMatch = re.search('^(g?[aikSw])(\\w+)', name)
        if nameMatch:
            yield (nameMatch.start(1), Keyword.Type, nameMatch.group(1))
            yield (nameMatch.start(2), Name, nameMatch.group(2))
        else:
            yield (match.start(), Name, name)