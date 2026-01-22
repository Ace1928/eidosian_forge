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
def get_random_id(length=6):
    return ''.join((random.choice(string.ascii_lowercase + string.digits) for _ in range(length)))