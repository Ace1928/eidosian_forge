import datetime
Parse RFC 822 dates and times
    http://tools.ietf.org/html/rfc822#section-5

    There are some formatting differences that are accounted for:
    1. Years may be two or four digits.
    2. The month and day can be swapped.
    3. Additional timezone names are supported.
    4. A default time and timezone are assumed if only a date is present.

    :param str date: a date/time string that will be converted to a time tuple
    :returns: a UTC time tuple, or None
    :rtype: time.struct_time | None
    