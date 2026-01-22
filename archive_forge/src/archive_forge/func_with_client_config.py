import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def with_client_config(self, client_config):
    """
        Create a copy with all original values and client-specific values.

        :type client_config: botocore.config.Config
        :param client_config: The client configuration object.
        """
    cp = copy(self)
    cp._client_config = client_config
    return cp