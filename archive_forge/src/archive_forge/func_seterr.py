import contextlib
import threading
def seterr(*, divide=None, over=None, under=None, invalid=None, linalg=None, fallback_mode=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = geterr()
    if divide is not None:
        raise NotImplementedError()
    if over is not None:
        raise NotImplementedError()
    if under is not None:
        raise NotImplementedError()
    if invalid is not None:
        raise NotImplementedError()
    if linalg is not None:
        if linalg in ('ignore', 'raise'):
            _config.linalg = linalg
        else:
            raise NotImplementedError()
    if fallback_mode is not None:
        if fallback_mode in ['print', 'warn', 'ignore', 'raise']:
            _config.fallback_mode = fallback_mode
        elif fallback_mode in ['log', 'call']:
            raise NotImplementedError
        else:
            raise ValueError('{} is not a valid dispatch type'.format(fallback_mode))
    _config.divide = divide
    _config.under = under
    _config.over = over
    _config.invalid = invalid
    return old_state