from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def save_as_html(self, path) -> None:
    if os.path.isdir(path):
        path = os.path.join(path, CARD_HTML_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(self.to_html())