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
def filtered_sequences():
    for seq in seqs:
        if not seq.artifacts:
            continue
        art = seq.artifacts[0]
        try:
            logged_by = _get_run_or_dummy_from_art(art, self.src_api)
        except requests.HTTPError as e:
            logger.error(f'Validate Artifact http error: art.entity={art.entity!r}, art.project={art.project!r}, art.name={art.name!r}, e={e!r}')
            continue
        if art.type == 'wandb-history' and isinstance(logged_by, _DummyRun):
            continue
        yield seq