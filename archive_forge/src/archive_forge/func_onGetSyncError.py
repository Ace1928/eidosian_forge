from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers.protocol_contacts.protocolentities  import GetSyncIqProtocolEntity, ResultSyncIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
import threading
import logging
def onGetSyncError(self, errorSyncIqProtocolEntity, originalIqProtocolEntity):
    print(errorSyncIqProtocolEntity)
    raise KeyboardInterrupt()