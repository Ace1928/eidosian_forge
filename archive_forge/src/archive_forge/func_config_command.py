import argparse
import os
from accelerate.utils import ComputeEnvironment
from .cluster import get_cluster_input
from .config_args import cache_dir, default_config_file, default_yaml_config_file, load_config_from_file  # noqa: F401
from .config_utils import _ask_field, _ask_options, _convert_compute_environment  # noqa: F401
from .sagemaker import get_sagemaker_input
def config_command(args):
    config = get_user_input()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        config_file = default_yaml_config_file
    if config_file.endswith('.json'):
        config.to_json_file(config_file)
    else:
        config.to_yaml_file(config_file)
    print(f'accelerate configuration saved at {config_file}')