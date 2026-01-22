from .message_media import MediaMessageProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message_meta import MessageMetaAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
@file_sha256.setter
def file_sha256(self, value):
    self.downloadablemedia_specific_attributes.file_sha256 = value