def order_error(subtag, got, expected):
    """
    Output an error indicating that tags were out of order.
    """
    options = SUBTAG_TYPES[expected:]
    if len(options) == 1:
        expect_str = options[0]
    elif len(options) == 2:
        expect_str = f'{options[0]} or {options[1]}'
    else:
        joined = ', '.join(options[:-1])
        last = options[-1]
        expect_str = f'{joined}, or {last}'
    got_str = SUBTAG_TYPES[got]
    raise LanguageTagError(f'This {got_str} subtag, {subtag!r}, is out of place. Expected {expect_str}.')