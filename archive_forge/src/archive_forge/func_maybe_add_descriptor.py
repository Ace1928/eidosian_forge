import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
@classmethod
def maybe_add_descriptor(cls, filename, package):
    descriptor = cls.registry.get(filename)
    if not descriptor:
        descriptor = cls.registry[filename] = cls(descriptor=descriptor_pb2.FileDescriptorProto(name=filename, package=package, syntax='proto3'), enums=collections.OrderedDict(), messages=collections.OrderedDict(), name=filename, nested={}, nested_enum={})
    return descriptor