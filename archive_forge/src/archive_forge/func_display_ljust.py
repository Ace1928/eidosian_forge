from unicodedata import normalize
from wcwidth import wcswidth, wcwidth
from ftfy.fixes import remove_terminal_escapes
def display_ljust(text, width, fillchar=' '):
    """
    Return `text` left-justified in a Unicode string whose display width,
    in a monospaced terminal, should be at least `width` character cells.
    The rest of the string will be padded with `fillchar`, which must be
    a width-1 character.

    "Left" here means toward the beginning of the string, which may actually
    appear on the right in an RTL context. This is similar to the use of the
    word "left" in "left parenthesis".

    >>> lines = ['Table flip', '(╯°□°)╯︵ ┻━┻', 'ちゃぶ台返し']
    >>> for line in lines:
    ...     print(display_ljust(line, 20, '▒'))
    Table flip▒▒▒▒▒▒▒▒▒▒
    (╯°□°)╯︵ ┻━┻▒▒▒▒▒▒▒
    ちゃぶ台返し▒▒▒▒▒▒▒▒

    This example, and the similar ones that follow, should come out justified
    correctly when viewed in a monospaced terminal. It will probably not look
    correct if you're viewing this code or documentation in a Web browser.
    """
    if character_width(fillchar) != 1:
        raise ValueError('The padding character must have display width 1')
    text_width = monospaced_width(text)
    if text_width == -1:
        return text
    padding = max(0, width - text_width)
    return text + fillchar * padding