from ..proto.wa20_pb2 import HandshakeMessage
@property
def server_hello(self):
    return self._server_hello