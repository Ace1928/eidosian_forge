from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
@classmethod
def raise_http_exception(cls, message: str, status_code: int=400, **kwargs) -> None:
    """
        Raises an HTTPException
        """
    raise HTTPException(status_code=status_code, detail=message, **kwargs)