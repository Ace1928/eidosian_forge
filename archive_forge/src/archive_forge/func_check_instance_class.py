from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
def check_instance_class(lexer, match, ctx):
    if match.group(0) == 'instance' or match.group(0) == 'class':
        yield (match.start(), Keyword, match.group(0))
    else:
        yield (match.start(), Name.Function, match.group(0))
        ctx.stack = ctx.stack + ['fromimportfunctype']
    ctx.pos = match.end()