import json
import os
import posixpath
import re
from hashlib import md5
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, storages
from django.utils.functional import LazyObject
def save_manifest(self):
    self.manifest_hash = self.file_hash(None, ContentFile(json.dumps(sorted(self.hashed_files.items())).encode()))
    payload = {'paths': self.hashed_files, 'version': self.manifest_version, 'hash': self.manifest_hash}
    if self.manifest_storage.exists(self.manifest_name):
        self.manifest_storage.delete(self.manifest_name)
    contents = json.dumps(payload).encode()
    self.manifest_storage._save(self.manifest_name, ContentFile(contents))