from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def deprecate_obj_name(obj: TObj, old_name: str, new_name: str, removal_date: str, include_st_prefix: bool=True) -> TObj:
    """Wrap an `st` object whose name has changed.

    Wrapped objects will behave as normal, but will also show an st.warning
    saying that the old name will be removed after `removal_date`.

    (We generally set `removal_date` to 3 months from the deprecation date.)

    Parameters
    ----------
    obj
        The `st.` object whose name has changed.

    old_name
        The object's deprecated name within __init__.py.

    new_name
        The object's new name within __init__.py.

    removal_date
        A date like "2020-01-01", indicating the last day we'll guarantee
        support for the deprecated name.

    include_st_prefix
        If False, does not prefix each of the object names in the deprecation
        essage with `st.*`. Defaults to True.
    """
    return _create_deprecated_obj_wrapper(obj, lambda: show_deprecation_warning(make_deprecated_name_warning(old_name, new_name, removal_date, include_st_prefix=include_st_prefix)))