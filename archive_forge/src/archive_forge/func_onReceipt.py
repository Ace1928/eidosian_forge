from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
@ProtocolEntityCallback('receipt')
def onReceipt(self, entity):
    self.toLower(entity.ack())