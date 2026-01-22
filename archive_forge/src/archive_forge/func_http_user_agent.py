import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def http_user_agent(user_agent: Union[Dict, str, None]=None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f'transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}'
    if is_torch_available():
        ua += f'; torch/{_torch_version}'
    if is_tf_available():
        ua += f'; tensorflow/{_tf_version}'
    if constants.HF_HUB_DISABLE_TELEMETRY:
        return ua + '; telemetry/off'
    if is_training_run_on_sagemaker():
        ua += '; ' + '; '.join((f'{k}/{v}' for k, v in define_sagemaker_information().items()))
    if os.environ.get('TRANSFORMERS_IS_CI', '').upper() in ENV_VARS_TRUE_VALUES:
        ua += '; is_ci/true'
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join((f'{k}/{v}' for k, v in user_agent.items()))
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua