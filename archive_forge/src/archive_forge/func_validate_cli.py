import sys
import warnings
from pathlib import Path
from typing import Tuple
import requests
from wasabi import Printer, msg
from .. import about
from ..util import (
from ._util import app
@app.command('validate')
def validate_cli():
    """
    Validate the currently installed pipeline packages and spaCy version. Checks
    if the installed packages are compatible and shows upgrade instructions if
    available. Should be run after `pip install -U spacy`.

    DOCS: https://spacy.io/api/cli#validate
    """
    validate()