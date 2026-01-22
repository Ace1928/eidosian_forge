import os
import random
import warnings
def insecureRandom(self, nbytes: int) -> bytes:
    """
        Return a number of non secure random bytes.

        @param nbytes: number of bytes to generate.
        @type nbytes: C{int}

        @return: a string of random bytes.
        @rtype: C{str}
        """
    try:
        return self._randBits(nbytes)
    except SourceNotAvailable:
        pass
    return self._randModule(nbytes)