def handleClash2(existing=[], prefix='', suffix=''):
    """
    existing should be a case-insensitive list
    of all existing file names.

    >>> prefix = ("0" * 5) + "."
    >>> suffix = "." + ("0" * 10)
    >>> existing = [prefix + str(i) + suffix for i in range(100)]

    >>> e = list(existing)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.100.0000000000')
    True

    >>> e = list(existing)
    >>> e.remove(prefix + "1" + suffix)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.1.0000000000')
    True

    >>> e = list(existing)
    >>> e.remove(prefix + "2" + suffix)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.2.0000000000')
    True
    """
    maxLength = maxFileNameLength - len(prefix) - len(suffix)
    maxValue = int('9' * maxLength)
    finalName = None
    counter = 1
    while finalName is None:
        fullName = prefix + str(counter) + suffix
        if fullName.lower() not in existing:
            finalName = fullName
            break
        else:
            counter += 1
        if counter >= maxValue:
            break
    if finalName is None:
        raise NameTranslationError('No unique name could be found.')
    return finalName