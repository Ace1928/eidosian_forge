import re
import markupsafe
def strip_ansi(source):
    """
    Remove ANSI escape codes from text.

    Parameters
    ----------
    source : str
        Source to remove the ANSI from

    """
    return _ANSI_RE.sub('', source)