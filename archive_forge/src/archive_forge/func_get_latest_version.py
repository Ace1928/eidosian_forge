import sys
from typing import Optional, Sequence
import requests
import typer
from wasabi import msg
from .. import about
from ..errors import OLD_MODEL_SHORTCUTS
from ..util import (
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app
def get_latest_version(model: str) -> str:
    comp = get_compatibility()
    return get_version(model, comp)