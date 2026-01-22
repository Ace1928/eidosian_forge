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
def should_emit_warning() -> bool:
    if get_all is False:
        return False
    if page is not None:
        return False
    if gl_list.per_page is None:
        return False
    if len(items) < gl_list.per_page:
        return False
    if gl_list.total is not None and len(items) >= gl_list.total:
        return False
    return True