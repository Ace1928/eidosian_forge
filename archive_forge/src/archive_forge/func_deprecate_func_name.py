from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def deprecate_func_name(func: TFunc, old_name: str, removal_date: str, extra_message: str | None=None, name_override: str | None=None) -> TFunc:
    """Wrap an `st` function whose name has changed.

    Wrapped functions will run as normal, but will also show an st.warning
    saying that the old name will be removed after removal_date.

    (We generally set `removal_date` to 3 months from the deprecation date.)

    Parameters
    ----------
    func
        The `st.` function whose name has changed.

    old_name
        The function's deprecated name within __init__.py.

    removal_date
        A date like "2020-01-01", indicating the last day we'll guarantee
        support for the deprecated name.

    extra_message
        An optional extra message to show in the deprecation warning.

    name_override
        An optional name to use in place of func.__name__.
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        show_deprecation_warning(make_deprecated_name_warning(old_name, name_override or func.__name__, removal_date, extra_message))
        return result
    wrapped_func.__name__ = old_name
    wrapped_func.__doc__ = func.__doc__
    return cast(TFunc, wrapped_func)