from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace', max_num_fields=None, separator='&'):
    """Parse a query given as a string argument.

        Arguments:

        qs: percent-encoded query string to be parsed

        keep_blank_values: flag indicating whether blank values in
            percent-encoded queries should be treated as blank strings.
            A true value indicates that blanks should be retained as blank
            strings.  The default false value indicates that blank values
            are to be ignored and treated as if they were  not included.

        strict_parsing: flag indicating what to do with parsing errors. If
            false (the default), errors are silently ignored. If true,
            errors raise a ValueError exception.

        encoding and errors: specify how to decode percent-encoded sequences
            into Unicode characters, as accepted by the bytes.decode() method.

        max_num_fields: int. If set, then throws a ValueError
            if there are more than n fields read by parse_qsl().

        separator: str. The symbol to use for separating the query arguments.
            Defaults to &.

        Returns a list, as G-d intended.
    """
    qs, _coerce_result = _coerce_args(qs)
    separator, _ = _coerce_args(separator)
    if not separator or not isinstance(separator, (str, bytes)):
        raise ValueError('Separator must be of type string or bytes.')
    if max_num_fields is not None:
        num_fields = 1 + qs.count(separator) if qs else 0
        if max_num_fields < num_fields:
            raise ValueError('Max number of fields exceeded')
    r = []
    query_args = qs.split(separator) if qs else []
    for name_value in query_args:
        if not name_value and (not strict_parsing):
            continue
        nv = name_value.split('=', 1)
        if len(nv) != 2:
            if strict_parsing:
                raise ValueError('bad query field: %r' % (name_value,))
            if keep_blank_values:
                nv.append('')
            else:
                continue
        if len(nv[1]) or keep_blank_values:
            name = nv[0].replace('+', ' ')
            name = unquote(name, encoding=encoding, errors=errors)
            name = _coerce_result(name)
            value = nv[1].replace('+', ' ')
            value = unquote(value, encoding=encoding, errors=errors)
            value = _coerce_result(value)
            r.append((name, value))
    return r