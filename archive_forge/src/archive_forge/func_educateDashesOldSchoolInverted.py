from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateDashesOldSchoolInverted(text):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with each instance of "--" translated to
                an em-dash character, and each "---" translated to
                an en-dash character. Two reasons why: First, unlike the
                en- and em-dash syntax supported by
                EducateDashesOldSchool(), it's compatible with existing
                entries written before SmartyPants 1.1, back when "--" was
                only used for em-dashes.  Second, em-dashes are more
                common than en-dashes, and so it sort of makes sense that
                the shortcut should be shorter to type. (Thanks to Aaron
                Swartz for the idea.)
    """
    text = re.sub('---', smartchars.endash, text)
    text = re.sub('--', smartchars.emdash, text)
    return text