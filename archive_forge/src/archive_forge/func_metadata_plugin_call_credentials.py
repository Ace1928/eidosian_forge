import collections
import logging
import threading
from typing import Callable, Optional, Type
import grpc
from grpc import _common
from grpc._cython import cygrpc
from grpc._typing import MetadataType
def metadata_plugin_call_credentials(metadata_plugin: grpc.AuthMetadataPlugin, name: Optional[str]) -> grpc.CallCredentials:
    if name is None:
        try:
            effective_name = metadata_plugin.__name__
        except AttributeError:
            effective_name = metadata_plugin.__class__.__name__
    else:
        effective_name = name
    return grpc.CallCredentials(cygrpc.MetadataPluginCallCredentials(_Plugin(metadata_plugin), _common.encode(effective_name)))