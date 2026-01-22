from __future__ import annotations
from typing import Optional
import grpc
from grpc._cython import cygrpc
from grpc._typing import MetadataType
def create_channel_option(compression: Optional[grpc.Compression]):
    return ((cygrpc.GRPC_COMPRESSION_CHANNEL_DEFAULT_ALGORITHM, int(compression)),) if compression else ()