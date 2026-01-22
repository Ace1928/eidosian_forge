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
def getStorageForProfile(profile_name):
    if type(profile_name) is not str:
        profile_name = str(profile_name)
    return StorageTools.constructPath(profile_name)