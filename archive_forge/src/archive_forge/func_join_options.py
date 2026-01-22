import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def join_options(options: t.Sequence[str]) -> t.Tuple[str, bool]:
    """Given a list of option strings this joins them in the most appropriate
    way and returns them in the form ``(formatted_string,
    any_prefix_is_slash)`` where the second item in the tuple is a flag that
    indicates if any of the option prefixes was a slash.
    """
    rv = []
    any_prefix_is_slash = False
    for opt in options:
        prefix = split_opt(opt)[0]
        if prefix == '/':
            any_prefix_is_slash = True
        rv.append((len(prefix), opt))
    rv.sort(key=lambda x: x[0])
    return (', '.join((x[1] for x in rv)), any_prefix_is_slash)