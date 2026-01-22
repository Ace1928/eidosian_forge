from suds.sax.enc import Encoder
@classmethod
def w3(cls, ns):
    try:
        return ns[1].startswith('http://www.w3.org')
    except Exception:
        pass
    return False