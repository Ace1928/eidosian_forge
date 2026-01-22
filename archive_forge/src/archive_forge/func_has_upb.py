import functools
def has_upb():
    try:
        from google._upb import _message
        has_upb = True
    except ImportError:
        has_upb = False
    return has_upb