import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import yaml
from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION
def load_config_from_file(config_file):
    if config_file is not None:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f'The passed configuration file `{config_file}` does not exist. Please pass an existing file to `accelerate launch`, or use the default one created through `accelerate config` and run `accelerate launch` without the `--config_file` argument.')
    else:
        config_file = default_config_file
    with open(config_file, encoding='utf-8') as f:
        if config_file.endswith('.json'):
            if json.load(f).get('compute_environment', ComputeEnvironment.LOCAL_MACHINE) == ComputeEnvironment.LOCAL_MACHINE:
                config_class = ClusterConfig
            else:
                config_class = SageMakerConfig
            return config_class.from_json_file(json_file=config_file)
        else:
            if yaml.safe_load(f).get('compute_environment', ComputeEnvironment.LOCAL_MACHINE) == ComputeEnvironment.LOCAL_MACHINE:
                config_class = ClusterConfig
            else:
                config_class = SageMakerConfig
            return config_class.from_yaml_file(yaml_file=config_file)