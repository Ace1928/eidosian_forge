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
def create_and_tag_model_card(repo_id: str, tags: Optional[List[str]]=None, token: Optional[str]=None, ignore_metadata_errors: bool=False):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """
    try:
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    except EntryNotFoundError:
        model_description = 'This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.'
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name='transformers')
        model_card = ModelCard.from_template(card_data, model_description=model_description)
    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)
    return model_card