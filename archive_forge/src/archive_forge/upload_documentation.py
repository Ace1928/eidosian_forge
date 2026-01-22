import os
import time
import warnings
from argparse import Namespace, _SubParsersAction
from typing import List, Optional
from huggingface_hub import logging
from huggingface_hub._commit_scheduler import CommitScheduler
from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RevisionNotFoundError, disable_progress_bars, enable_progress_bars
Contains command to upload a repo or file with the CLI.

Usage:
    # Upload file (implicit)
    huggingface-cli upload my-cool-model ./my-cool-model.safetensors

    # Upload file (explicit)
    huggingface-cli upload my-cool-model ./my-cool-model.safetensors  model.safetensors

    # Upload directory (implicit). If `my-cool-model/` is a directory it will be uploaded, otherwise an exception is raised.
    huggingface-cli upload my-cool-model

    # Upload directory (explicit)
    huggingface-cli upload my-cool-model ./models/my-cool-model .

    # Upload filtered directory (example: tensorboard logs except for the last run)
    huggingface-cli upload my-cool-model ./model/training /logs --include "*.tfevents.*" --exclude "*20230905*"

    # Upload private dataset
    huggingface-cli upload Wauplin/my-cool-dataset ./data . --repo-type=dataset --private

    # Upload with token
    huggingface-cli upload Wauplin/my-cool-model --token=hf_****

    # Sync local Space with Hub (upload new files, delete removed files)
    huggingface-cli upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"

    # Schedule commits every 30 minutes
    huggingface-cli upload Wauplin/my-cool-model --every=30
