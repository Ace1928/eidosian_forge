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
@ProtocolEntityCallback('stream:error')
def onStreamError(self, streamErrorEntity):
    logger.error(streamErrorEntity)
    if self.getProp(self.__class__.PROP_RECONNECT_ON_STREAM_ERR, True):
        if streamErrorEntity.getErrorType() == StreamErrorProtocolEntity.TYPE_CONFLICT:
            logger.warn('Not reconnecting because you signed in in another location')
        else:
            logger.info('Initiating reconnect')
            self.reconnect = True
    else:
        logger.warn('Not reconnecting because property %s is not set' % self.__class__.PROP_RECONNECT_ON_STREAM_ERR)
    self.toUpper(streamErrorEntity)
    self.disconnect()