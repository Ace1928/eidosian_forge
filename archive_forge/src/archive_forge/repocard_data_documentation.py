import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
Format the internal data dict. In this case, we convert eval results to a valid model index