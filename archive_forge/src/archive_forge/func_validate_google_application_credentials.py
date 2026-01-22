import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
@validator('google_application_credentials')
def validate_google_application_credentials(cls, v):
    if v is None:
        return pathlib.Path.home().joinpath('adc.json')
    if _fileio_available:
        return File(v)
    if isinstance(v, str):
        v = pathlib.Path(v)
    return v