import warnings
from typing import Dict, Optional, Sequence
def unidecode_expect_ascii(string: str, errors: str='ignore', replace_str: str='?') -> str:
    """Transliterate an Unicode object into an ASCII string

    >>> unidecode("北亰")
    "Bei Jing "

    This function first tries to convert the string using ASCII codec.
    If it fails (because of non-ASCII characters), it falls back to
    transliteration using the character tables.

    This is approx. five times faster if the string only contains ASCII
    characters, but slightly slower than unicode_expect_nonascii if
    non-ASCII characters are present.

    errors specifies what to do with characters that have not been
    found in replacement tables. The default is 'ignore' which ignores
    the character. 'strict' raises an UnidecodeError. 'replace'
    substitutes the character with replace_str (default is '?').
    'preserve' keeps the original character.

    Note that if 'preserve' is used the returned string might not be
    ASCII!
    """
    try:
        bytestring = string.encode('ASCII')
    except UnicodeEncodeError:
        pass
    else:
        return string
    return _unidecode(string, errors, replace_str)