import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_latin_ligatures(text):
    """
    Replace single-character ligatures of Latin letters, such as 'ﬁ', with the
    characters that they contain, as in 'fi'. Latin ligatures are usually not
    intended in text strings (though they're lovely in *rendered* text).  If
    you have such a ligature in your string, it is probably a result of a
    copy-and-paste glitch.

    We leave ligatures in other scripts alone to be safe. They may be intended,
    and removing them may lose information. If you want to take apart nearly
    all ligatures, use NFKC normalization.

        >>> print(fix_latin_ligatures("ﬂuﬃeﬆ"))
        fluffiest
    """
    return text.translate(LIGATURES)