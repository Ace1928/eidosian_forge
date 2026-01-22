from pyasn1 import error
def getValues(self, *names):
    try:
        return [self.__names[name] for name in names]
    except KeyError:
        raise error.PyAsn1Error('Unknown bit identifier(s): %s' % (set(names).difference(self.__names),))