import contextlib
import threading
def get_config_linalg():
    try:
        value = _config.linalg
    except AttributeError:
        value = _config.linalg = 'ignore'
    return value