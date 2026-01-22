import argparse
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from shutil import which
from typing import Any, Dict, List, Tuple
import torch
from ..commands.config.config_args import SageMakerConfig
from ..utils import (
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import is_port_in_use, merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType
def prepare_sagemager_args_inputs(sagemaker_config: SageMakerConfig, args: argparse.Namespace) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    print('Configuring Amazon SageMaker environment')
    os.environ['AWS_DEFAULT_REGION'] = sagemaker_config.region
    if sagemaker_config.profile is not None:
        os.environ['AWS_PROFILE'] = sagemaker_config.profile
    elif args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
    else:
        raise OSError('You need to provide an aws_access_key_id and aws_secret_access_key when not using aws_profile')
    source_dir = os.path.dirname(args.training_script)
    if not source_dir:
        source_dir = '.'
    entry_point = os.path.basename(args.training_script)
    if not entry_point.endswith('.py'):
        raise ValueError(f'Your training script should be a python script and not "{entry_point}"')
    print('Converting Arguments to Hyperparameters')
    hyperparameters = _convert_nargs_to_dict(args.training_script_args)
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(f'Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}.')
    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f'Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}.')
    environment = {'ACCELERATE_USE_SAGEMAKER': 'true', 'ACCELERATE_MIXED_PRECISION': str(mixed_precision), 'ACCELERATE_DYNAMO_BACKEND': dynamo_backend.value, 'ACCELERATE_DYNAMO_MODE': args.dynamo_mode, 'ACCELERATE_DYNAMO_USE_FULLGRAPH': str(args.dynamo_use_fullgraph), 'ACCELERATE_DYNAMO_USE_DYNAMIC': str(args.dynamo_use_dynamic), 'ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE': sagemaker_config.distributed_type.value}
    distribution = None
    if sagemaker_config.distributed_type == SageMakerDistributedType.DATA_PARALLEL:
        distribution = {'smdistributed': {'dataparallel': {'enabled': True}}}
    sagemaker_inputs = None
    if sagemaker_config.sagemaker_inputs_file is not None:
        print(f'Loading SageMaker Inputs from {sagemaker_config.sagemaker_inputs_file} file')
        sagemaker_inputs = {}
        with open(sagemaker_config.sagemaker_inputs_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split('\t')
                sagemaker_inputs[l[0]] = l[1].strip()
        print(f'Loaded SageMaker Inputs: {sagemaker_inputs}')
    sagemaker_metrics = None
    if sagemaker_config.sagemaker_metrics_file is not None:
        print(f'Loading SageMaker Metrics from {sagemaker_config.sagemaker_metrics_file} file')
        sagemaker_metrics = []
        with open(sagemaker_config.sagemaker_metrics_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split('\t')
                metric_dict = {'Name': l[0], 'Regex': l[1].strip()}
                sagemaker_metrics.append(metric_dict)
        print(f'Loaded SageMaker Metrics: {sagemaker_metrics}')
    print('Creating Estimator')
    args = {'image_uri': sagemaker_config.image_uri, 'entry_point': entry_point, 'source_dir': source_dir, 'role': sagemaker_config.iam_role_name, 'transformers_version': sagemaker_config.transformers_version, 'pytorch_version': sagemaker_config.pytorch_version, 'py_version': sagemaker_config.py_version, 'base_job_name': sagemaker_config.base_job_name, 'instance_count': sagemaker_config.num_machines, 'instance_type': sagemaker_config.ec2_instance_type, 'debugger_hook_config': False, 'distribution': distribution, 'hyperparameters': hyperparameters, 'environment': environment, 'metric_definitions': sagemaker_metrics}
    if sagemaker_config.additional_args is not None:
        args = merge_dicts(sagemaker_config.additional_args, args)
    return (args, sagemaker_inputs)