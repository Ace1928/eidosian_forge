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
def location_to_proto(self, location_attributes):
    location_message = Message.LocationMessage()
    if location_attributes.degrees_latitude is not None:
        location_message.degrees_latitude = location_attributes.degrees_latitude
    if location_attributes.degrees_longitude is not None:
        location_message.degrees_longitude = location_attributes.degrees_longitude
    if location_attributes.name is not None:
        location_message.name = location_attributes.name
    if location_attributes.address is not None:
        location_message.address = location_attributes.address
    if location_attributes.url is not None:
        location_message.url = location_attributes.url
    if location_attributes.duration is not None:
        location_message.duration = location_attributes.duration
    if location_attributes.accuracy_in_meters is not None:
        location_message.accuracy_in_meters = location_attributes.accuracy_in_meters
    if location_attributes.speed_in_mps is not None:
        location_message.speed_in_mps = location_attributes.speed_in_mps
    if location_attributes.degrees_clockwise_from_magnetic_north is not None:
        location_message.degrees_clockwise_from_magnetic_north = location_attributes.degrees_clockwise_from_magnetic_north
    if location_attributes.axolotl_sender_key_distribution_message is not None:
        location_message._axolotl_sender_key_distribution_message = location_attributes.axolotl_sender_key_distribution_message
    if location_attributes.jpeg_thumbnail is not None:
        location_message.jpeg_thumbnail = location_attributes.jpeg_thumbnail
    return location_message