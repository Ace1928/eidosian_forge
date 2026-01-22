import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def check_uwsgi_thread_support():
    try:
        from uwsgi import opt
    except ImportError:
        return True
    from sentry_sdk.consts import FALSE_VALUES

    def enabled(option):
        value = opt.get(option, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, bytes):
            try:
                value = value.decode()
            except Exception:
                pass
        return value and str(value).lower() not in FALSE_VALUES
    threads_enabled = 'threads' in opt or enabled('enable-threads')
    fork_hooks_on = enabled('py-call-uwsgi-fork-hooks')
    lazy_mode = enabled('lazy-apps') or enabled('lazy')
    if lazy_mode and (not threads_enabled):
        from warnings import warn
        warn(Warning('IMPORTANT: We detected the use of uWSGI without thread support. This might lead to unexpected issues. Please run uWSGI with "--enable-threads" for full support.'))
        return False
    elif not lazy_mode and (not threads_enabled or not fork_hooks_on):
        from warnings import warn
        warn(Warning('IMPORTANT: We detected the use of uWSGI in preforking mode without thread support. This might lead to crashing workers. Please run uWSGI with both "--enable-threads" and "--py-call-uwsgi-fork-hooks" for full support.'))
        return False
    return True