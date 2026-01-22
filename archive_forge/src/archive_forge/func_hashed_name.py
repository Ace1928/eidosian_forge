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
def hashed_name(self, name, content=None, filename=None):
    parsed_name = urlsplit(unquote(name))
    clean_name = parsed_name.path.strip()
    filename = filename and urlsplit(unquote(filename)).path.strip() or clean_name
    opened = content is None
    if opened:
        if not self.exists(filename):
            raise ValueError("The file '%s' could not be found with %r." % (filename, self))
        try:
            content = self.open(filename)
        except OSError:
            return name
    try:
        file_hash = self.file_hash(clean_name, content)
    finally:
        if opened:
            content.close()
    path, filename = os.path.split(clean_name)
    root, ext = os.path.splitext(filename)
    file_hash = '.%s' % file_hash if file_hash else ''
    hashed_name = os.path.join(path, '%s%s%s' % (root, file_hash, ext))
    unparsed_name = list(parsed_name)
    unparsed_name[2] = hashed_name
    if '?#' in name and (not unparsed_name[3]):
        unparsed_name[2] += '?'
    return urlunsplit(unparsed_name)