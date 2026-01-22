from __future__ import annotations
from .base import BaseModel
from pydantic import Field, PrivateAttr
from ..utils.lazy import logger
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
def get_app_update_counts(self) -> Dict[str, int]:
    """
        Returns the update counts
        """
    return {'allowed_origins': len(self.allowed_origins), 'callbacks': len(self.callbacks), 'web_origins': len(self.web_origins), 'allowed_logout_urls': len(self.allowed_logout_urls)}