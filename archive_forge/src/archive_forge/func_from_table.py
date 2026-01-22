import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
@classmethod
def from_table(cls, table_name: str, chart_fields: dict=None, chart_strings: dict=None):
    return cls(query={'summaryTable': {'tableKey': table_name}}, chart_fields=chart_fields, chart_strings=chart_strings)