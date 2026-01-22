import abc
from enum import Enum
import os
from google.auth import _helpers, environment_vars
from google.auth import exceptions
from google.auth import metrics
from google.auth._refresh_worker import RefreshThreadManager
def with_non_blocking_refresh(self):
    self._use_non_blocking_refresh = True