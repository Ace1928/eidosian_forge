import os
import pathlib
from django.core.exceptions import SuspiciousFileOperation
from django.core.files import File
from django.core.files.utils import validate_file_name
from django.utils.crypto import get_random_string
from django.utils.text import get_valid_filename
def get_available_name(self, name, max_length=None):
    """
        Return a filename that's free on the target storage system and
        available for new content to be written to.
        """
    name = str(name).replace('\\', '/')
    dir_name, file_name = os.path.split(name)
    if '..' in pathlib.PurePath(dir_name).parts:
        raise SuspiciousFileOperation("Detected path traversal attempt in '%s'" % dir_name)
    validate_file_name(file_name)
    file_root, file_ext = os.path.splitext(file_name)
    while self.exists(name) or (max_length and len(name) > max_length):
        name = os.path.join(dir_name, self.get_alternative_name(file_root, file_ext))
        if max_length is None:
            continue
        truncation = len(name) - max_length
        if truncation > 0:
            file_root = file_root[:-truncation]
            if not file_root:
                raise SuspiciousFileOperation('Storage can not find an available filename for "%s". Please make sure that the corresponding file field allows sufficient "max_length".' % name)
            name = os.path.join(dir_name, self.get_alternative_name(file_root, file_ext))
    return name