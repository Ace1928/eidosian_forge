from __future__ import annotations
import json
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any
from gradio_client import media_data, utils
from gradio_client.data_classes import FileData
def serialized_info(self):
    return {'type': 'string', 'description': 'path to directory with images and a file associating images with captions called captions.json'}