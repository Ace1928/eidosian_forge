import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
@property
def last_modified_datetime(self) -> Optional[datetime]:
    """
        :type: datetime
        """
    return self._makeHttpDatetimeAttribute(self.last_modified).value