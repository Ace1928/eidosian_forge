from unicodedata import normalize
from wcwidth import wcswidth, wcwidth
from ftfy.fixes import remove_terminal_escapes
def monospaced_width(text: str) -> int:
    """
    Return the number of character cells that this string is likely to occupy
    when displayed in a monospaced, modern, Unicode-aware terminal emulator.
    We refer to this as the "display width" of the string.

    This can be useful for formatting text that may contain non-spacing
    characters, or CJK characters that take up two character cells.

    Returns -1 if the string contains a non-printable or control character.

    >>> monospaced_width('ちゃぶ台返し')
    12
    >>> len('ちゃぶ台返し')
    6
    >>> monospaced_width('owl\\N{SOFT HYPHEN}flavored')
    11
    >>> monospaced_width('example\\x80')
    -1

    A more complex example: The Korean word 'ibnida' can be written with 3
    pre-composed characters or 7 jamo. Either way, it *looks* the same and
    takes up 6 character cells.

    >>> monospaced_width('입니다')
    6
    >>> monospaced_width('\\u110b\\u1175\\u11b8\\u1102\\u1175\\u1103\\u1161')
    6

    The word "blue" with terminal escapes to make it blue still takes up only
    4 characters, when shown as intended.
    >>> monospaced_width('\\x1b[34mblue\\x1b[m')
    4
    """
    return int(wcswidth(remove_terminal_escapes(normalize('NFC', text))))