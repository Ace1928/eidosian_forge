from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.protocol_messages.proto.protocol_pb2 import MessageKey
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_image import ImageAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_downloadablemedia \
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_media import MediaAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_context_info import ContextInfoAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
from yowsup.layers.protocol_messages.proto.e2e_pb2 import ContextInfo
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_extendedtext import ExtendedTextAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_document import DocumentAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_contact import ContactAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_location import LocationAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_video import VideoAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_audio import AudioAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_sticker import StickerAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_sender_key_distribution_message import \
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_protocol import ProtocolAttributes
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_protocol import MessageKeyAttributes
def downloadablemedia_to_proto(self, downloadablemedia_attributes, proto):
    proto.mimetype = downloadablemedia_attributes.mimetype
    proto.file_length = downloadablemedia_attributes.file_length
    proto.file_sha256 = downloadablemedia_attributes.file_sha256
    if downloadablemedia_attributes.url is not None:
        proto.url = downloadablemedia_attributes.url
    if downloadablemedia_attributes.media_key is not None:
        proto.media_key = downloadablemedia_attributes.media_key
    return self.media_to_proto(downloadablemedia_attributes, proto)