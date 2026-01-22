from pyasn1 import error
@property
def presentTypes(self):
    """Return *TagSet* to ASN.1 type map present in callee *TagMap*"""
    return self.__presentTypes