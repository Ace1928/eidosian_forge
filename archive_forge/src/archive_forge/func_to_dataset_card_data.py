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
def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
    if self:
        if 'dataset_info' in dataset_card_data and isinstance(dataset_card_data['dataset_info'], dict):
            dataset_metadata_infos = {dataset_card_data['dataset_info'].get('config_name', 'default'): dataset_card_data['dataset_info']}
        elif 'dataset_info' in dataset_card_data and isinstance(dataset_card_data['dataset_info'], list):
            dataset_metadata_infos = {config_metadata['config_name']: config_metadata for config_metadata in dataset_card_data['dataset_info']}
        else:
            dataset_metadata_infos = {}
        total_dataset_infos = {**dataset_metadata_infos, **{config_name: dset_info._to_yaml_dict() for config_name, dset_info in self.items()}}
        for config_name, dset_info_yaml_dict in total_dataset_infos.items():
            dset_info_yaml_dict['config_name'] = config_name
        if len(total_dataset_infos) == 1:
            dataset_card_data['dataset_info'] = next(iter(total_dataset_infos.values()))
            config_name = dataset_card_data['dataset_info'].pop('config_name', None)
            if config_name != 'default':
                dataset_card_data['dataset_info'] = {'config_name': config_name, **dataset_card_data['dataset_info']}
        else:
            dataset_card_data['dataset_info'] = []
            for config_name, dataset_info_yaml_dict in sorted(total_dataset_infos.items()):
                dataset_info_yaml_dict.pop('config_name', None)
                dataset_info_yaml_dict = {'config_name': config_name, **dataset_info_yaml_dict}
                dataset_card_data['dataset_info'].append(dataset_info_yaml_dict)