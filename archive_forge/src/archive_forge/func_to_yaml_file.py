import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import yaml
from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION
def to_yaml_file(self, yaml_file):
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(self.to_dict(), f)