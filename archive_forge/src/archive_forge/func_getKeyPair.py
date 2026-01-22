from .storageprotos_pb2 import SignedPreKeyRecordStructure
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
def getKeyPair(self):
    publicKey = Curve.decodePoint(bytearray(self.structure.publicKey), 0)
    privateKey = Curve.decodePrivatePoint(bytearray(self.structure.privateKey))
    return ECKeyPair(publicKey, privateKey)