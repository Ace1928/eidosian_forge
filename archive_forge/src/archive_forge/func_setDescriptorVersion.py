import math
def setDescriptorVersion(version='1.0.0'):
    """ Set the version on the descriptor function.

  Use as a decorator """

    def wrapper(func):
        func.version = version
        return func
    return wrapper