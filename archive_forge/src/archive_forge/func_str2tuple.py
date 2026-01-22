def str2tuple(s, sep='/'):
    """
    Given the string representation of a tagged token, return the
    corresponding tuple representation.  The rightmost occurrence of
    *sep* in *s* will be used to divide *s* into a word string and
    a tag string.  If *sep* does not occur in *s*, return (s, None).

        >>> from nltk.tag.util import str2tuple
        >>> str2tuple('fly/NN')
        ('fly', 'NN')

    :type s: str
    :param s: The string representation of a tagged token.
    :type sep: str
    :param sep: The separator string used to separate word strings
        from tags.
    """
    loc = s.rfind(sep)
    if loc >= 0:
        return (s[:loc], s[loc + len(sep):].upper())
    else:
        return (s, None)