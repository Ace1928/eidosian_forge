import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
def isEndOfStream(substrate):
    """Check whether we have reached the end of a stream.

    Although it is more effective to read and catch exceptions, this
    function

    Parameters
    ----------
    substrate: :py:class:`IOBase`
        Stream to check

    Returns
    -------
    : :py:class:`bool`
    """
    if isinstance(substrate, io.BytesIO):
        cp = substrate.tell()
        substrate.seek(0, os.SEEK_END)
        result = substrate.tell() == cp
        substrate.seek(cp, os.SEEK_SET)
        yield result
    else:
        received = substrate.read(1)
        if received is None:
            yield
        if received:
            substrate.seek(-1, os.SEEK_CUR)
        yield (not received)