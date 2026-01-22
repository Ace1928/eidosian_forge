import os
from .constants import YowConstants
import codecs, sys
import logging
import tempfile
import base64
import hashlib
import os.path, mimetypes
import uuid
from consonance.structs.keypair import KeyPair
from appdirs import user_config_dir
from .optionalmodules import PILOptionalModule, FFVideoOptionalModule
@staticmethod
def generatePreviewFromImage(image):
    fd, path = tempfile.mkstemp()
    preview = None
    if ImageTools.scaleImage(image, path, 'JPEG', YowConstants.PREVIEW_WIDTH, YowConstants.PREVIEW_HEIGHT):
        fileObj = os.fdopen(fd, 'rb+')
        fileObj.seek(0)
        preview = fileObj.read()
        fileObj.close()
    os.remove(path)
    return preview