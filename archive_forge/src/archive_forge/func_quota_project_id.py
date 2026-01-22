import abc
import os
import six
from google.auth import _helpers, environment_vars
from google.auth import exceptions
@property
def quota_project_id(self):
    """Project to use for quota and billing purposes."""
    return self._quota_project_id