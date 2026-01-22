from yowsup.layers.protocol_messages.protocolentities.protomessage import ProtomessageProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
import logging
@media_type.setter
def media_type(self, value):
    if value not in MediaMessageProtocolEntity.TYPES_MEDIA:
        logger.warn("media type: '%s' is not supported" % value)
    self._media_type = value