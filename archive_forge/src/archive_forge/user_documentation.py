import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand

    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    