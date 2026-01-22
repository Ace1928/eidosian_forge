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
def next_page(self) -> Optional[int]:
    """The next page number.

        If None, the current page is the last.
        """
    return int(self._next_page) if self._next_page else None