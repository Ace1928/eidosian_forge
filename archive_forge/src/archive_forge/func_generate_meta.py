import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import srsly
from catalogue import RegistryError
from thinc.api import Config
from wasabi import MarkdownRenderer, Printer, get_raw_input
from .. import about, util
from ..compat import importlib_metadata
from ..schemas import ModelMetaSchema, validate
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app, string_to_list
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
def generate_meta(existing_meta: Dict[str, Any], msg: Printer) -> Dict[str, Any]:
    meta = existing_meta or {}
    settings = [('lang', 'Pipeline language', meta.get('lang', 'en')), ('name', 'Pipeline name', meta.get('name', 'pipeline')), ('version', 'Package version', meta.get('version', '0.0.0')), ('description', 'Package description', meta.get('description', None)), ('author', 'Author', meta.get('author', None)), ('email', 'Author email', meta.get('email', None)), ('url', 'Author website', meta.get('url', None)), ('license', 'License', meta.get('license', 'MIT'))]
    msg.divider('Generating meta.json')
    msg.text('Enter the package settings for your pipeline. The following information will be read from your pipeline data: pipeline, vectors.')
    for setting, desc, default in settings:
        response = get_raw_input(desc, default)
        meta[setting] = default if response == '' and default else response
    return meta