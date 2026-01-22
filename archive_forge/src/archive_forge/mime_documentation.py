import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound

    Lexer for Multipurpose Internet Mail Extensions (MIME) data. This lexer is
    designed to process nested multipart data.

    It assumes that the given data contains both header and body (and is
    split at an empty line). If no valid header is found, then the entire data
    will be treated as body.

    Additional options accepted:

    `MIME-max-level`
        Max recursion level for nested MIME structure. Any negative number
        would treated as unlimited. (default: -1)

    `Content-Type`
        Treat the data as a specific content type. Useful when header is
        missing, or this lexer would try to parse from header. (default:
        `text/plain`)

    `Multipart-Boundary`
        Set the default multipart boundary delimiter. This option is only used
        when `Content-Type` is `multipart` and header is missing. This lexer
        would try to parse from header by default. (default: None)

    `Content-Transfer-Encoding`
        Treat the data as a specific encoding. Or this lexer would try to parse
        from header by default. (default: None)

    .. versionadded:: 2.5
    