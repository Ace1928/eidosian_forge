import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
def content_callback(self, match):
    content_type = getattr(self, 'content_type', None)
    content = match.group()
    offset = match.start()
    if content_type:
        from pygments.lexers import get_lexer_for_mimetype
        possible_lexer_mimetypes = [content_type]
        if '+' in content_type:
            general_type = re.sub('^(.*)/.*\\+(.*)$', '\\1/\\2', content_type)
            possible_lexer_mimetypes.append(general_type)
        for i in possible_lexer_mimetypes:
            try:
                lexer = get_lexer_for_mimetype(i)
            except ClassNotFound:
                pass
            else:
                for idx, token, value in lexer.get_tokens_unprocessed(content):
                    yield (offset + idx, token, value)
                return
    yield (offset, Text, content)