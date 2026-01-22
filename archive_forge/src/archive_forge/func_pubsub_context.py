from base64 import b64encode
from typing import Mapping, Optional, NamedTuple
import logging
import pkg_resources
from cloudsdk.google.protobuf import struct_pb2  # pytype: disable=pyi-error
def pubsub_context(framework: Optional[str]=None) -> Mapping[str, str]:
    """Construct the pubsub context mapping for the given framework."""
    context = struct_pb2.Struct()
    context.fields['language'].string_value = 'PYTHON'
    if framework:
        context.fields['framework'].string_value = framework
    version = _version()
    context.fields['major_version'].number_value = version.major
    context.fields['minor_version'].number_value = version.minor
    encoded = b64encode(context.SerializeToString()).decode('utf-8')
    return {'x-goog-pubsub-context': encoded}