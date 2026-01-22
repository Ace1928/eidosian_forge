from __future__ import annotations
import logging # isort:skip
import os
from pathlib import Path
from tornado.web import HTTPError, StaticFileHandler
from ...core.types import PathLike
def validate_absolute_path(self, root: dict[str, PathLike], absolute_path: str) -> str | None:
    for artifacts_dir in root.values():
        if Path(absolute_path).is_relative_to(artifacts_dir):
            return super().validate_absolute_path(str(artifacts_dir), absolute_path)
    return None