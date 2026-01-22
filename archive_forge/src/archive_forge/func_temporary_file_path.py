import os
from io import BytesIO
from django.conf import settings
from django.core.files import temp as tempfile
from django.core.files.base import File
from django.core.files.utils import validate_file_name
def temporary_file_path(self):
    """Return the full path of this file."""
    return self.file.name