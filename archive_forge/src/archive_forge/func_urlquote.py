import sys
import types
def urlquote(s, safe='/'):
    if isinstance(s, _real_unicode):
        s = s.encode('utf8')
    return _urlquote(s, safe)