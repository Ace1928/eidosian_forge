import sys
def deleter(self, fdel):
    result = type(self)(self.fget, self.fset, fdel, self.__doc__)
    result.overwrite_doc = self.overwrite_doc
    return result