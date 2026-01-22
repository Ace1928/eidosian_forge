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
def extendedtext_to_proto(self, extendedtext_attributes):
    m = Message.ExtendedTextMessage()
    if extendedtext_attributes.text is not None:
        m.text = extendedtext_attributes.text
    if extendedtext_attributes.matched_text is not None:
        m.matched_text = extendedtext_attributes.matched_text
    if extendedtext_attributes.canonical_url is not None:
        m.canonical_url = extendedtext_attributes.canonical_url
    if extendedtext_attributes.description is not None:
        m.description = extendedtext_attributes.description
    if extendedtext_attributes.title is not None:
        m.title = extendedtext_attributes.title
    if extendedtext_attributes.jpeg_thumbnail is not None:
        m.jpeg_thumbnail = extendedtext_attributes.jpeg_thumbnail
    if extendedtext_attributes.context_info is not None:
        m.context_info.MergeFrom(self.contextinfo_to_proto(extendedtext_attributes.context_info))
    return m