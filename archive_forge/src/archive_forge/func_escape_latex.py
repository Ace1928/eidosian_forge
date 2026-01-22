import re
def escape_latex(text):
    """
    Escape characters that may conflict with latex.

    Parameters
    ----------
    text : str
        Text containing characters that may conflict with Latex
    """
    text = ''.join((LATEX_SUBS.get(c, c) for c in text))
    for pattern, replacement in LATEX_RE_SUBS:
        text = pattern.sub(replacement, text)
    return text