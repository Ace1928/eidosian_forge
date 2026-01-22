import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
def detect_platform():
    """Returns the platform and device information."""
    if on_gcp():
        if context.context().list_logical_devices('GPU'):
            return PlatformDevice.GCE_GPU
        elif context.context().list_logical_devices('TPU'):
            return PlatformDevice.GCE_TPU
        else:
            return PlatformDevice.GCE_CPU
    elif context.context().list_logical_devices('GPU'):
        return PlatformDevice.INTERNAL_GPU
    elif context.context().list_logical_devices('TPU'):
        return PlatformDevice.INTERNAL_TPU
    else:
        return PlatformDevice.INTERNAL_CPU