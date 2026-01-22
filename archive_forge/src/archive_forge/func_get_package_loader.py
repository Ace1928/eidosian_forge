from __future__ import annotations
import importlib.util
import mimetypes
import os
import posixpath
import typing as t
from datetime import datetime
from datetime import timezone
from io import BytesIO
from time import time
from zlib import adler32
from ..http import http_date
from ..http import is_resource_modified
from ..security import safe_join
from ..utils import get_content_type
from ..wsgi import get_path_info
from ..wsgi import wrap_file
def get_package_loader(self, package: str, package_path: str) -> _TLoader:
    load_time = datetime.now(timezone.utc)
    spec = importlib.util.find_spec(package)
    reader = spec.loader.get_resource_reader(package)

    def loader(path: str | None) -> tuple[str | None, _TOpener | None]:
        if path is None:
            return (None, None)
        path = safe_join(package_path, path)
        if path is None:
            return (None, None)
        basename = posixpath.basename(path)
        try:
            resource = reader.open_resource(path)
        except OSError:
            return (None, None)
        if isinstance(resource, BytesIO):
            return (basename, lambda: (resource, load_time, len(resource.getvalue())))
        return (basename, lambda: (resource, datetime.fromtimestamp(os.path.getmtime(resource.name), tz=timezone.utc), os.path.getsize(resource.name)))
    return loader