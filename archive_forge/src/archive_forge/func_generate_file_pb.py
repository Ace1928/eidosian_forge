import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
def generate_file_pb(self, new_class, fallback_salt=''):
    """Generate the descriptors for all protos in the file.

        This method takes the file descriptor attached to the parent
        message and generates the immutable descriptors for all of the
        messages in the file descriptor. (This must be done in one fell
        swoop for immutability and to resolve proto cross-referencing.)

        This is run automatically when the last proto in the file is
        generated, as determined by the module's __all__ tuple.
        """
    pool = descriptor_pool.Default()
    salt = self._calculate_salt(new_class, fallback_salt)
    self.descriptor.name = '{name}.proto'.format(name='_'.join([self.descriptor.name[:-6], salt]).rstrip('_'))
    pool.Add(self.descriptor)
    for full_name, proto_plus_message in self.messages.items():
        descriptor = pool.FindMessageTypeByName(full_name)
        pb_message = reflection.GeneratedProtocolMessageType(descriptor.name, (message.Message,), {'DESCRIPTOR': descriptor, '__module__': None})
        proto_plus_message._meta._pb = pb_message
        proto_plus_message._meta.marshal.register(pb_message, MessageRule(pb_message, proto_plus_message))
        for field in proto_plus_message._meta.fields.values():
            if field.message and isinstance(field.message, str):
                field.message = self.messages[field.message]
            elif field.enum and isinstance(field.enum, str):
                field.enum = self.enums[field.enum]
    for full_name, proto_plus_enum in self.enums.items():
        descriptor = pool.FindEnumTypeByName(full_name)
        proto_plus_enum._meta.pb = descriptor
    self.registry.pop(self.name)