from __future__ import annotations
import typing as t
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Request as RequestBase
from werkzeug.wrappers import Response as ResponseBase
from . import json
from .globals import current_app
from .helpers import _split_blueprint_path
@property
def max_content_length(self) -> int | None:
    """Read-only view of the ``MAX_CONTENT_LENGTH`` config key."""
    if current_app:
        return current_app.config['MAX_CONTENT_LENGTH']
    else:
        return None