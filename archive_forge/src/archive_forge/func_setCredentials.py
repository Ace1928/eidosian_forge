from yowsup.layers import YowLayerInterface
def setCredentials(self, phone, keypair):
    self._layer.setCredentials((phone, keypair))