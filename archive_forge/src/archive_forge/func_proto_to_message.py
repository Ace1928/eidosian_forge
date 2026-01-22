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
def proto_to_message(self, proto):
    conversation = proto.conversation if proto.conversation else None
    image = self.proto_to_image(proto.image_message) if proto.HasField('image_message') else None
    contact = self.proto_to_contact(proto.contact_message) if proto.HasField('contact_message') else None
    location = self.proto_to_location(proto.location_message) if proto.HasField('location_message') else None
    extended_text = self.proto_to_extendedtext(proto.extended_text_message) if proto.HasField('extended_text_message') else None
    document = self.proto_to_document(proto.document_message) if proto.HasField('document_message') else None
    audio = self.proto_to_audio(proto.audio_message) if proto.HasField('audio_message') else None
    video = self.proto_to_video(proto.video_message) if proto.HasField('video_message') else None
    sticker = self.proto_to_sticker(proto.sticker_message) if proto.HasField('sticker_message') else None
    sender_key_distribution_message = self.proto_to_sender_key_distribution_message(proto.sender_key_distribution_message) if proto.HasField('sender_key_distribution_message') else None
    protocol = self.proto_to_protocol(proto.protocol_message) if proto.HasField('protocol_message') else None
    return MessageAttributes(conversation, image, contact, location, extended_text, document, audio, video, sticker, sender_key_distribution_message, protocol)