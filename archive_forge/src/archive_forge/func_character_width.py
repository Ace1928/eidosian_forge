from unicodedata import normalize
from wcwidth import wcswidth, wcwidth
from ftfy.fixes import remove_terminal_escapes
def character_width(char: str) -> int:
    """
    Determine the width that a character is likely to be displayed as in
    a monospaced terminal. The width for a printable character will
    always be 0, 1, or 2.

    Nonprintable or control characters will return -1, a convention that comes
    from wcwidth.

    >>> character_width('車')
    2
    >>> character_width('A')
    1
    >>> character_width('\\N{ZERO WIDTH JOINER}')
    0
    >>> character_width('\\n')
    -1
    """
    return int(wcwidth(char))