from unittest import TestCase
from zmq.utils import z85
def test_client_secret(self):
    client_secret = b'{\xb8d\xb4\x89\xaf\xa3g\x1f\xbei\x10\x1f\x94\xb3\x89r\xf2H\x16\xdf\xb0\x1bQek?\xec\x8d\xfd\x08\x88'
    encoded = z85.encode(client_secret)
    assert encoded == b'D:)Q[IlAW!ahhC2ac:9*A}h:p?([4%wOTJ%JR%cs'
    decoded = z85.decode(encoded)
    assert decoded == client_secret