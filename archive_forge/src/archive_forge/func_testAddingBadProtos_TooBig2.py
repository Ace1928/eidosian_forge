from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def testAddingBadProtos_TooBig2(self) -> None:
    """Adding a protocol with a number >=2**32 raises an exception."""
    e = ip.IPProtocol()
    try:
        e.addProto(2 ** 32 + 1, MyProtocol([]))
    except TypeError as e:
        if e.args == ('Added protocol must fit in 32 bits',):
            pass
        else:
            raise
    else:
        raise AssertionError('addProto must raise an exception for bad protocols')