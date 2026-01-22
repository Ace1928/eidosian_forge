from __future__ import annotations
import time
from ..types.properties import StatefulProperty
from ..types.user_session import UserSession
from ..utils.lazy import logger
from .admin import AZManagementClient
from typing import List, Optional, Any, Dict, TYPE_CHECKING
@property
def session_data(self) -> Dict[str, Any]:
    """
        Returns the Session User Data
        """
    return self.load_data()