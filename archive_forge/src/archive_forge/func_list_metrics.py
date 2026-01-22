import inspect
import os
import shutil
import warnings
from pathlib import Path, PurePath
from typing import Dict, List, Mapping, Optional, Sequence, Union
import huggingface_hub
from . import config
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager
from .info import DatasetInfo
from .load import (
from .utils.deprecation_utils import deprecated
from .utils.file_utils import relative_to_absolute_path
from .utils.logging import get_logger
from .utils.version import Version
@deprecated("Use 'evaluate.list_evaluation_modules' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate")
def list_metrics(with_community_metrics=True, with_details=False):
    """List all the metrics script available on the Hugging Face Hub.

    <Deprecated version="2.5.0">

    Use `evaluate.list_evaluation_modules` instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate

    </Deprecated>

    Args:
        with_community_metrics (:obj:`bool`, optional, default ``True``): Include the community provided metrics.
        with_details (:obj:`bool`, optional, default ``False``): Return the full details on the metrics instead of only the short name.

    Example:

    ```py
    >>> from datasets import list_metrics
    >>> list_metrics()
    ['accuracy',
     'bertscore',
     'bleu',
     'bleurt',
     'cer',
     'chrf',
     ...
    ]
    ```
    """
    metrics = huggingface_hub.list_metrics()
    if not with_community_metrics:
        metrics = [metric for metric in metrics if '/' not in metric.id]
    if not with_details:
        metrics = [metric.id for metric in metrics]
    return metrics