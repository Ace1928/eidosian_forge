from enum import EnumMeta
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from proto.primitives import ProtoType
@property
def descriptor(self):
    """Return the descriptor for the field."""
    if not self._descriptor:
        type_name = None
        if isinstance(self.message, str):
            if not self.message.startswith(self.package):
                self.message = '{package}.{name}'.format(package=self.package, name=self.message)
            type_name = self.message
        elif self.message:
            type_name = self.message.DESCRIPTOR.full_name if hasattr(self.message, 'DESCRIPTOR') else self.message._meta.full_name
        elif isinstance(self.enum, str):
            if not self.enum.startswith(self.package):
                self.enum = '{package}.{name}'.format(package=self.package, name=self.enum)
            type_name = self.enum
        elif self.enum:
            type_name = self.enum.DESCRIPTOR.full_name if hasattr(self.enum, 'DESCRIPTOR') else self.enum._meta.full_name
        self._descriptor = descriptor_pb2.FieldDescriptorProto(name=self.name, number=self.number, label=3 if self.repeated else 1, type=self.proto_type, type_name=type_name, json_name=self.json_name, proto3_optional=self.optional)
    return self._descriptor