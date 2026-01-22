import re
from pygments.lexer import Lexer
from pygments.token import Token, Error, Text
from pygments.util import get_choice_opt, text_type, BytesIO

    Recreate a token stream formatted with the `RawTokenFormatter`.  This
    lexer raises exceptions during parsing if the token stream in the
    file is malformed.

    Additional options accepted:

    `compress`
        If set to ``"gz"`` or ``"bz2"``, decompress the token stream with
        the given compression algorithm before lexing (default: ``""``).
    