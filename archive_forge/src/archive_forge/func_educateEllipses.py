from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateEllipses(text):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with each instance of "..." translated to
                an ellipsis character.

    Example input:  Huh...?
    Example output: Huh&#8230;?
    """
    text = re.sub('\\.\\.\\.', smartchars.ellipsis, text)
    text = re.sub('\\. \\. \\.', smartchars.ellipsis, text)
    return text