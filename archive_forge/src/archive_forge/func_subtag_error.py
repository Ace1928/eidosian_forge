def subtag_error(subtag, expected='a valid subtag'):
    """
    Try to output a reasonably helpful error message based on our state of
    parsing. Most of this code is about how to list, in English, the kinds
    of things we were expecting to find.
    """
    raise LanguageTagError(f'Expected {expected}, got {subtag!r}')