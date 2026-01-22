import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub.utils import logging, yaml_dump
@property
def unique_identifier(self) -> tuple:
    """Returns a tuple that uniquely identifies this evaluation."""
    return (self.task_type, self.dataset_type, self.dataset_config, self.dataset_split, self.dataset_revision)