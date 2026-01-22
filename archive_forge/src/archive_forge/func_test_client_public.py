from unittest import TestCase
from zmq.utils import z85
def test_client_public(self):
    client_public = b'\xbb\x88G\x1de\xe2e\x9b0\xc5ZS!\xce\xbbZ\xab+p\xa3\x98d\\&\xdc\xa2\xb2\xfc\xb4?\xc5\x18'
    encoded = z85.encode(client_public)
    assert encoded == b'Yne@$w-vo<fVvi]a<NY6T1ed:M$fCG*[IaLV{hID'
    decoded = z85.decode(encoded)
    assert decoded == client_public