import unittest
from wsme import WSRoot
from wsme.protocol import getprotocol, CallContext, Protocol
import wsme.protocol
def test_register_protocol(self):
    wsme.protocol.register_protocol(DummyProtocol)
    assert wsme.protocol.registered_protocols['dummy'] == DummyProtocol
    r = WSRoot()
    assert len(r.protocols) == 0
    r.addprotocol('dummy')
    assert len(r.protocols) == 1
    assert r.protocols[0].__class__ == DummyProtocol
    r = WSRoot(['dummy'])
    assert len(r.protocols) == 1
    assert r.protocols[0].__class__ == DummyProtocol