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
def cancel_delegation_token(self, token):
    """Stop the token from being useful"""
    self._call('CANCELDELEGATIONTOKEN', method='put', token=token)