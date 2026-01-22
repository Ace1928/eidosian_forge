from suds.sax.enc import Encoder
@classmethod
def isns(cls, ns):
    try:
        return isinstance(ns, tuple) and len(ns) == len(cls.default)
    except Exception:
        pass
    return False