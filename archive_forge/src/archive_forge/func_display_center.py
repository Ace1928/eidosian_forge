from unicodedata import normalize
from wcwidth import wcswidth, wcwidth
from ftfy.fixes import remove_terminal_escapes
def display_center(text, width, fillchar=' '):
    """
    Return `text` centered in a Unicode string whose display width, in a
    monospaced terminal, should be at least `width` character cells. The rest
    of the string will be padded with `fillchar`, which must be a width-1
    character.

    >>> lines = ['Table flip', '(╯°□°)╯︵ ┻━┻', 'ちゃぶ台返し']
    >>> for line in lines:
    ...     print(display_center(line, 20, '▒'))
    ▒▒▒▒▒Table flip▒▒▒▒▒
    ▒▒▒(╯°□°)╯︵ ┻━┻▒▒▒▒
    ▒▒▒▒ちゃぶ台返し▒▒▒▒
    """
    if character_width(fillchar) != 1:
        raise ValueError('The padding character must have display width 1')
    text_width = monospaced_width(text)
    if text_width == -1:
        return text
    padding = max(0, width - text_width)
    left_padding = padding // 2
    right_padding = padding - left_padding
    return fillchar * left_padding + text + fillchar * right_padding