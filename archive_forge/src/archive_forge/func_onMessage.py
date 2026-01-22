from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
@ProtocolEntityCallback('message')
def onMessage(self, messageProtocolEntity):
    if messageProtocolEntity.getType() == 'text':
        self.onTextMessage(messageProtocolEntity)
    elif messageProtocolEntity.getType() == 'media':
        self.onMediaMessage(messageProtocolEntity)
    self.toLower(messageProtocolEntity.forward(messageProtocolEntity.getFrom()))
    self.toLower(messageProtocolEntity.ack())
    self.toLower(messageProtocolEntity.ack(True))