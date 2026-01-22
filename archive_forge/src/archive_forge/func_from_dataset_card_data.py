import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
@classmethod
def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> 'DatasetInfosDict':
    if isinstance(dataset_card_data.get('dataset_info'), (list, dict)):
        if isinstance(dataset_card_data['dataset_info'], list):
            return cls({dataset_info_yaml_dict.get('config_name', 'default'): DatasetInfo._from_yaml_dict(dataset_info_yaml_dict) for dataset_info_yaml_dict in dataset_card_data['dataset_info']})
        else:
            dataset_info = DatasetInfo._from_yaml_dict(dataset_card_data['dataset_info'])
            dataset_info.config_name = dataset_card_data['dataset_info'].get('config_name', 'default')
            return cls({dataset_info.config_name: dataset_info})
    else:
        return cls()