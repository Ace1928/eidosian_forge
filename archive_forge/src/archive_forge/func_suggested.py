from __future__ import annotations
import difflib
import typing as t
from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect
@cached_property
def suggested(self) -> Rule | None:
    return self.closest_rule(self.adapter)