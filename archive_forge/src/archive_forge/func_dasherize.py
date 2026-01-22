import re
import unicodedata
def dasherize(word):
    """Replace underscores with dashes in the string.

    Example::

        >>> dasherize("puni_puni")
        'puni-puni'

    """
    return word.replace('_', '-')