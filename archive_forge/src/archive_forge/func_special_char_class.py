import collections
import io
import random as random_module
import string
def special_char_class(allowed_chars, min_count=0):
    """Return a character class containing custom characters.

    The result of this function can be passed to :func:`generate_password` as
    one of the character classes to use in generating a password.

    :param allowed_chars: Iterable of the characters in the character class
    :param min_count: The minimum number of members of this class to appear in
                      a generated password
    """
    return CharClass(frozenset(allowed_chars), min_count)