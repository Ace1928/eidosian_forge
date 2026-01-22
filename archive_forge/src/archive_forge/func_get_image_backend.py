import os
import warnings
from modulefinder import Module
import torch
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .extension import _HAS_OPS
def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend