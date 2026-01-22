from pygments.lexer import RegexLexer, bygroups, do_insertions, include, \
from pygments.token import Comment, Error, Keyword, Name, Number, Operator, \
from pygments.util import ClassNotFound, get_bool_opt
def handle_annotated_strings(self, match):
    """Adds syntax from another languages inside annotated strings

        match args:
            1:open_string,
            2:exclamation_mark,
            3:lang_name,
            4:space_or_newline,
            5:code,
            6:close_string
        """
    from pygments.lexers import get_lexer_by_name
    yield (match.start(1), String.Double, match.group(1))
    yield (match.start(2), String.Interpol, match.group(2))
    yield (match.start(3), String.Interpol, match.group(3))
    yield (match.start(4), Text.Whitespace, match.group(4))
    lexer = None
    if self.handle_annotateds:
        try:
            lexer = get_lexer_by_name(match.group(3).strip())
        except ClassNotFound:
            pass
    code = match.group(5)
    if lexer is None:
        yield (match.group(5), String, code)
    else:
        yield from do_insertions([], lexer.get_tokens_unprocessed(code))
    yield (match.start(6), String.Double, match.group(6))