import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def remove_control_chars(text):
    """
    Remove various control characters that you probably didn't intend to be in
    your text. Many of these characters appear in the table of "Characters not
    suitable for use with markup" at
    http://www.unicode.org/reports/tr20/tr20-9.html.

    This includes:

    - ASCII control characters, except for the important whitespace characters
      (U+00 to U+08, U+0B, U+0E to U+1F, U+7F)
    - Deprecated Arabic control characters (U+206A to U+206F)
    - Interlinear annotation characters (U+FFF9 to U+FFFB)
    - The Object Replacement Character (U+FFFC)
    - The byte order mark (U+FEFF)

    However, these similar characters are left alone:

    - Control characters that produce whitespace (U+09, U+0A, U+0C, U+0D,
      U+2028, and U+2029)
    - C1 control characters (U+80 to U+9F) -- even though they are basically
      never used intentionally, they are important clues about what mojibake
      has happened
    - Control characters that affect glyph rendering, such as joiners and
      right-to-left marks (U+200C to U+200F, U+202A to U+202E)
    - Musical notation control characters (U+1D173 to U+1D17A) because wow if
      you're using those you probably have a good reason
    - Tag characters, because they are now used in emoji sequences such as
      "Flag of Wales"
    """
    return text.translate(CONTROL_CHARS)