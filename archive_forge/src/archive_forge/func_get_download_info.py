from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def get_download_info(self, material_ids, task_types=None, file_patterns=None):
    """Get a list of URLs to retrieve raw VASP output files from the NoMaD repository.

        Args:
            material_ids (list): list of material identifiers (mp-id's)
            task_types (list): list of task types to include in download (see TaskType Enum class)
            file_patterns (list): list of wildcard file names to include for each task

        Returns:
            a tuple of 1) a dictionary mapping material_ids to task_ids and
            task_types, and 2) a list of URLs to download zip archives from
            NoMaD repository. Each zip archive will contain a manifest.json with
            metadata info, e.g. the task/external_ids that belong to a directory
        """
    task_types = [typ.value for typ in task_types or [] if isinstance(typ, TaskType)]
    meta = {}
    for doc in self.query({'task_id': {'$in': material_ids}}, ['task_id', 'blessed_tasks']):
        for task_type, task_id in doc['blessed_tasks'].items():
            if task_types and task_type not in task_types:
                continue
            mp_id = doc['task_id']
            if meta.get(mp_id) is None:
                meta[mp_id] = [{'task_id': task_id, 'task_type': task_type}]
            else:
                meta[mp_id].append({'task_id': task_id, 'task_type': task_type})
    if not meta:
        raise ValueError(f'No tasks found for material id {material_ids}.')
    prefix = 'https://nomad-lab.eu/prod/rae/api/repo/?'
    if file_patterns is not None:
        for file_pattern in file_patterns:
            prefix += f'file_pattern={file_pattern!r}&'
    prefix += 'external_id='
    task_ids = [task['task_id'] for task_list in meta.values() for task in task_list]
    nomad_exist_task_ids = self._check_get_download_info_url_by_task_id(prefix=prefix, task_ids=task_ids)
    if len(nomad_exist_task_ids) != len(task_ids):
        self._print_help_message(nomad_exist_task_ids, task_ids, file_patterns, task_types)
    prefix = 'https://nomad-lab.eu/prod/rae/api/raw/query?'
    if file_patterns is not None:
        for file_pattern in file_patterns:
            prefix += f'file_pattern={file_pattern}&'
    prefix += 'external_id='
    urls = [f'{prefix}{task_ids}' for task_ids in nomad_exist_task_ids]
    return (meta, urls)