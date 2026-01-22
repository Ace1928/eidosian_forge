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
def output_api_info(self) -> tuple[str, str]:
    api_info = self.api_info()
    types = api_info.get('serialized_output', [api_info['info']['type']] * 2)
    return (types[0], types[1])