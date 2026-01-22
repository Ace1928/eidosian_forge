import re
from pygments.lexer import ExtendedRegexLexer, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def blob_callback(lexer, match, ctx):
    text_before_blob = match.group(1)
    blob_start = match.group(2)
    blob_size_str = match.group(3)
    blob_size = int(blob_size_str)
    yield (match.start(), String, text_before_blob)
    ctx.pos += len(text_before_blob)
    if ctx.text[match.end() + blob_size] != ')':
        result = '\\B(' + blob_size_str + ')('
        yield (match.start(), String, result)
        ctx.pos += len(result)
        return
    blob_text = blob_start + ctx.text[match.end():match.end() + blob_size] + ')'
    yield (match.start(), String.Escape, blob_text)
    ctx.pos = match.end() + blob_size + 1