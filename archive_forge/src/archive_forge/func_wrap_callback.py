import functools
import logging
def wrap_callback(func):
    """Wrap a callback generator so it fails safely"""

    @functools.wraps(func)
    def safe_progress(*args, **kwargs):

        def callback(*args, **kwargs):
            return None
        if not allow_show_progress():
            return callback
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _disable_progress(e)
            return callback
    return safe_progress