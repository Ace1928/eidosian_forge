import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def notebook_metadata(silent: bool) -> Dict[str, str]:
    """Attempt to query jupyter for the path and name of the notebook file.

    This can handle different jupyter environments, specifically:

    1. Colab
    2. Kaggle
    3. JupyterLab
    4. Notebooks
    5. Other?
    """
    error_message = 'Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.'
    try:
        jupyter_metadata = notebook_metadata_from_jupyter_servers_and_kernel_id()
        ipynb = attempt_colab_load_ipynb()
        if ipynb is not None and jupyter_metadata is not None:
            return {'root': '/content', 'path': jupyter_metadata['path'], 'name': jupyter_metadata['name']}
        if wandb.util._is_kaggle():
            ipynb = attempt_kaggle_load_ipynb()
            if ipynb:
                return {'root': '/kaggle/working', 'path': ipynb['metadata']['name'], 'name': ipynb['metadata']['name']}
        if jupyter_metadata:
            return jupyter_metadata
        if not silent:
            logger.error(error_message)
        return {}
    except Exception:
        if not silent:
            logger.error(error_message)
        return {}