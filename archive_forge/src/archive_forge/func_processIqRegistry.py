from yowsup.layers import YowLayer, YowLayerEvent
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_media.protocolentities.iq_requestupload import RequestUploadIqProtocolEntity
from yowsup.layers.protocol_media.mediauploader import MediaUploader
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers.auth.protocolentities import StreamErrorProtocolEntity
from yowsup.layers import EventCallback
import inspect
import logging
def processIqRegistry(self, entity):
    """
        :type entity: IqProtocolEntity
        """
    if entity.getTag() == 'iq':
        iq_id = entity.getId()
        if iq_id in self.iqRegistry:
            originalIq, successClbk, errorClbk = self.iqRegistry[iq_id]
            del self.iqRegistry[iq_id]
            if entity.getType() == IqProtocolEntity.TYPE_RESULT and successClbk:
                successClbk(entity, originalIq)
            elif entity.getType() == IqProtocolEntity.TYPE_ERROR and errorClbk:
                errorClbk(entity, originalIq)
            return True
    return False