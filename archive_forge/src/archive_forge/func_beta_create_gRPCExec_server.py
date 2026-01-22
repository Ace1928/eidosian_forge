from __future__ import absolute_import, division, print_function
import sys
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper
import grpc
from grpc.beta import implementations as beta_implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities
def beta_create_gRPCExec_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    request_deserializers = {('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdJSONOutput'): ShowCmdArgs.FromString, ('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdTextOutput'): ShowCmdArgs.FromString}
    response_serializers = {('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdJSONOutput'): ShowCmdJSONReply.SerializeToString, ('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdTextOutput'): ShowCmdTextReply.SerializeToString}
    method_implementations = {('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdJSONOutput'): face_utilities.unary_stream_inline(servicer.ShowCmdJSONOutput), ('IOSXRExtensibleManagabilityService.gRPCExec', 'ShowCmdTextOutput'): face_utilities.unary_stream_inline(servicer.ShowCmdTextOutput)}
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)