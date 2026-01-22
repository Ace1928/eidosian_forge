from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@property
def default_client_name(self):
    """
        Returns the default API client name
        """
    if self.in_k8s:
        return 'cluster'
    return 'local' if self.client.api_dev_mode else 'external'