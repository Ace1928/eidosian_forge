from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateDashesOldSchool(text):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with each instance of "--" translated to
                an en-dash character, and each "---" translated to
                an em-dash character.
    """
    text = re.sub('---', smartchars.emdash, text)
    text = re.sub('--', smartchars.endash, text)
    return text