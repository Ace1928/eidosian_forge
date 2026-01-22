from suds.sax.enc import Encoder
@classmethod
def xsi(cls, ns):
    try:
        return cls.w3(ns) and ns[1].endswith('XMLSchema-instance')
    except Exception:
        pass
    return False