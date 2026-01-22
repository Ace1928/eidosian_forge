from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
def save_image(self, image):
    """Save to temporary file and return filename."""
    return image._dump(format=self.get_format(image), **self.options)