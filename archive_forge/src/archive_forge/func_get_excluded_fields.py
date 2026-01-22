from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def get_excluded_fields(self, exclude: Optional[Set[str]]=None) -> Set[str]:
    """
        Returns the excluded fields
        """
    exclude = exclude or set()
    exclude.update(self._excluded_fields)
    return exclude