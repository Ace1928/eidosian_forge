from .message import MessageProtocolEntity
from .proto import ProtoProtocolEntity
from yowsup.layers.protocol_messages.protocolentities.attributes.converter import AttributesConverter
from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_message import MessageAttributes
import logging
@message_attributes.setter
def message_attributes(self, value):
    self._message_attributes = value