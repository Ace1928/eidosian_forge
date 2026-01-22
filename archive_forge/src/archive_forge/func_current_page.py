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
def current_page(self) -> int:
    """The current page number."""
    if TYPE_CHECKING:
        assert self._current_page is not None
    return int(self._current_page)