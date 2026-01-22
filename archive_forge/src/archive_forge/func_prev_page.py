import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@property
def prev_page(self) -> Optional[int]:
    """The previous page number.

        If None, the current page is the first.
        """
    return int(self._prev_page) if self._prev_page else None