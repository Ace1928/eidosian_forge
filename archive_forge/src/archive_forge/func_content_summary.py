import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote
import requests
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize
def content_summary(self, path):
    """Total numbers of files, directories and bytes under path"""
    out = self._call('GETCONTENTSUMMARY', path=path)
    return out.json()['ContentSummary']