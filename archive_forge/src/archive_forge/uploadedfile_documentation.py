import os
from io import BytesIO
from django.conf import settings
from django.core.files import temp as tempfile
from django.core.files.base import File
from django.core.files.utils import validate_file_name

        Create a SimpleUploadedFile object from a dictionary with keys:
           - filename
           - content-type
           - content
        