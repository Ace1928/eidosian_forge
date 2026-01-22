import warnings
from collections import namedtuple
def test_aifc(h, f):
    """AIFC and AIFF files"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        import aifc
    if not h.startswith(b'FORM'):
        return None
    if h[8:12] == b'AIFC':
        fmt = 'aifc'
    elif h[8:12] == b'AIFF':
        fmt = 'aiff'
    else:
        return None
    f.seek(0)
    try:
        a = aifc.open(f, 'r')
    except (EOFError, aifc.Error):
        return None
    return (fmt, a.getframerate(), a.getnchannels(), a.getnframes(), 8 * a.getsampwidth())