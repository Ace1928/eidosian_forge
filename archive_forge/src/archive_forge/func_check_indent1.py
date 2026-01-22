from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
def check_indent1(lexer, match, ctx):
    indent, reallen = CleanLexer.indent_len(match.group(0))
    if indent > ctx.indent:
        yield (match.start(), Whitespace, match.group(0))
        ctx.pos = match.start() + reallen + 1
    else:
        ctx.indent = 0
        ctx.pos = match.start()
        ctx.stack = ctx.stack[:-1]
        yield (match.start(), Whitespace, match.group(0)[1:])