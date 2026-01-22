import logging
import re
from typing import (
from . import settings
from .utils import choplist
def do_keyword(self, pos: int, token: PSKeyword) -> None:
    return