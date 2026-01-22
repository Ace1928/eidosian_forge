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
def renew_delegation_token(self, token):
    """Make token live longer. Returns new expiry time"""
    out = self._call('RENEWDELEGATIONTOKEN', method='put', token=token)
    return out.json()['long']