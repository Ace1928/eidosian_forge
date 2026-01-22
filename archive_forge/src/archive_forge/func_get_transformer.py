from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
def get_transformer(transform, **rpc_options):
    """Return the appropriate transformer class"""
    if transform is None:
        raise ValueError('Invalid transform')
    if isinstance(transform, Affine):
        transformer_cls = partial(AffineTransformer, transform)
    elif isinstance(transform, RPC):
        transformer_cls = partial(RPCTransformer, transform, **rpc_options)
    else:
        transformer_cls = partial(GCPTransformer, transform)
    return transformer_cls