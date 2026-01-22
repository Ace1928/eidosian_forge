import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
def import_all(self, *, runs: bool=True, artifacts: bool=True, reports: bool=True, namespaces: Optional[Iterable[Namespace]]=None, incremental: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
    logger.info(f'START: Importing all, runs={runs!r}, artifacts={artifacts!r}, reports={reports!r}')
    if runs:
        self.import_runs(namespaces=namespaces, incremental=incremental, remapping=remapping)
    if reports:
        self.import_reports(namespaces=namespaces, remapping=remapping)
    if artifacts:
        self.import_artifact_sequences(namespaces=namespaces, incremental=incremental, remapping=remapping)
    logger.info('END: Importing all')