from unicodedata import normalize
from wcwidth import wcswidth, wcwidth
from ftfy.fixes import remove_terminal_escapes
def display_rjust(text, width, fillchar=' '):
    """
    Return `text` right-justified in a Unicode string whose display width,
    in a monospaced terminal, should be at least `width` character cells.
    The rest of the string will be padded with `fillchar`, which must be
    a width-1 character.

    "Right" here means toward the end of the string, which may actually be on
    the left in an RTL context. This is similar to the use of the word "right"
    in "right parenthesis".

    >>> lines = ['Table flip', '(╯°□°)╯︵ ┻━┻', 'ちゃぶ台返し']
    >>> for line in lines:
    ...     print(display_rjust(line, 20, '▒'))
    ▒▒▒▒▒▒▒▒▒▒Table flip
    ▒▒▒▒▒▒▒(╯°□°)╯︵ ┻━┻
    ▒▒▒▒▒▒▒▒ちゃぶ台返し
    """
    if character_width(fillchar) != 1:
        raise ValueError('The padding character must have display width 1')
    text_width = monospaced_width(text)
    if text_width == -1:
        return text
    padding = max(0, width - text_width)
    return fillchar * padding + text