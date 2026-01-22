import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
from zipfile import ZipFile, ZipInfo
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
from pip._internal.utils.wheel import parse_wheel
def get_csv_rows_for_installed(old_csv_rows: List[List[str]], installed: Dict[RecordPath, RecordPath], changed: Set[RecordPath], generated: List[str], lib_dir: str) -> List[InstalledCSVRow]:
    """
    :param installed: A map from archive RECORD path to installation RECORD
        path.
    """
    installed_rows: List[InstalledCSVRow] = []
    for row in old_csv_rows:
        if len(row) > 3:
            logger.warning('RECORD line has more than three elements: %s', row)
        old_record_path = cast('RecordPath', row[0])
        new_record_path = installed.pop(old_record_path, old_record_path)
        if new_record_path in changed:
            digest, length = rehash(_record_to_fs_path(new_record_path, lib_dir))
        else:
            digest = row[1] if len(row) > 1 else ''
            length = row[2] if len(row) > 2 else ''
        installed_rows.append((new_record_path, digest, length))
    for f in generated:
        path = _fs_to_record_path(f, lib_dir)
        digest, length = rehash(f)
        installed_rows.append((path, digest, length))
    return installed_rows + [(installed_record_path, '', '') for installed_record_path in installed.values()]