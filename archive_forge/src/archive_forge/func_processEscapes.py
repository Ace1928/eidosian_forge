from a single quote by the algorithm. Therefore, a text like::
import re, sys
def processEscapes(text, restore=False):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with after processing the following backslash
                escape sequences. This is useful if you want to force a "dumb"
                quote or other character to appear.

                Escape  Value
                ------  -----
                \\\\      &#92;
                \\"      &#34;
                \\'      &#39;
                \\.      &#46;
                \\-      &#45;
                \\`      &#96;
    """
    replacements = (('\\\\', '&#92;'), ('\\"', '&#34;'), ("\\'", '&#39;'), ('\\.', '&#46;'), ('\\-', '&#45;'), ('\\`', '&#96;'))
    if restore:
        for ch, rep in replacements:
            text = text.replace(rep, ch[1])
    else:
        for ch, rep in replacements:
            text = text.replace(ch, rep)
    return text