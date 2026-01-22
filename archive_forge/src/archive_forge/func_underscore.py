import re
import unicodedata
def underscore(word):
    """
    Make an underscored, lowercase form from the expression in the string.

    Example::

        >>> underscore("DeviceType")
        'device_type'

    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::

        >>> camelize(underscore("IOError"))
        'IoError'

    """
    word = re.sub('([A-Z]+)([A-Z][a-z])', '\\1_\\2', word)
    word = re.sub('([a-z\\d])([A-Z])', '\\1_\\2', word)
    word = word.replace('-', '_')
    return word.lower()