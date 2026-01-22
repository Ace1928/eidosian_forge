from __future__ import annotations
from pydantic import model_validator
from lazyops.types import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
def lookup_id(self, name: str) -> Optional[str]:
    """
        Lookup ID
        """
    return self.uids.get(name, self.users.get(name, self.channels.get(name)))