from __future__ import annotations
import contextlib
import threading
from typing import Iterator
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
@contextlib.contextmanager
def spinner(text: str='In progress...', *, _cache: bool=False) -> Iterator[None]:
    """Temporarily displays a message while executing a block of code.

    Parameters
    ----------
    text : str
        A message to display while executing that block

    Example
    -------

    >>> import time
    >>> import streamlit as st
    >>>
    >>> with st.spinner('Wait for it...'):
    >>>     time.sleep(5)
    >>> st.success('Done!')

    """
    import streamlit.runtime.caching as caching
    import streamlit.runtime.legacy_caching.caching as legacy_caching
    from streamlit.proto.Spinner_pb2 import Spinner as SpinnerProto
    from streamlit.string_util import clean_text
    with legacy_caching.suppress_cached_st_function_warning():
        with caching.suppress_cached_st_function_warning():
            message = st.empty()
    DELAY_SECS = 0.5
    display_message = True
    display_message_lock = threading.Lock()
    try:

        def set_message():
            with display_message_lock:
                if display_message:
                    with legacy_caching.suppress_cached_st_function_warning():
                        with caching.suppress_cached_st_function_warning():
                            spinner_proto = SpinnerProto()
                            spinner_proto.text = clean_text(text)
                            spinner_proto.cache = _cache
                            message._enqueue('spinner', spinner_proto)
        add_script_run_ctx(threading.Timer(DELAY_SECS, set_message)).start()
        yield
    finally:
        if display_message_lock:
            with display_message_lock:
                display_message = False
        with legacy_caching.suppress_cached_st_function_warning():
            with caching.suppress_cached_st_function_warning():
                if 'chat_message' in set(message._active_dg._ancestor_block_types):
                    message.container()
                else:
                    message.empty()