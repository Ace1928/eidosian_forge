import os
import warnings
from modulefinder import Module
import torch
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
def set_video_backend(backend):
    """
    Specifies the package used to decode videos.

    Args:
        backend (string): Name of the video backend. one of {'pyav', 'video_reader'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It generally decodes faster than :mod:`pyav`, but is perhaps less robust.

    .. note::
        Building with FFMPEG is disabled by default in the latest `main`. If you want to use the 'video_reader'
        backend, please compile torchvision from source.
    """
    global _video_backend
    if backend not in ['pyav', 'video_reader', 'cuda']:
        raise ValueError("Invalid video backend '%s'. Options are 'pyav', 'video_reader' and 'cuda'" % backend)
    if backend == 'video_reader' and (not io._HAS_VIDEO_OPT):
        message = 'video_reader video backend is not available. Please compile torchvision from source and try again'
        raise RuntimeError(message)
    elif backend == 'cuda' and (not io._HAS_GPU_VIDEO_DECODER):
        message = 'cuda video backend is not available.'
        raise RuntimeError(message)
    else:
        _video_backend = backend