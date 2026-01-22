import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import yaml
def start_monitor(config_file: str):
    cluster_config = _read_yaml(config_file)
    provider_config = cluster_config['provider']
    assert provider_config['type'] == 'fake_multinode_docker', f'The docker monitor only works with providers of type `fake_multinode_docker`, got `{provider_config['type']}`'
    project_name = provider_config['project_name']
    volume_dir = provider_config['shared_volume_dir']
    os.makedirs(volume_dir, mode=493, exist_ok=True)
    bootstrap_config_path = os.path.join(volume_dir, 'bootstrap_config.yaml')
    shutil.copy(config_file, bootstrap_config_path)
    docker_compose_config_path = os.path.join(volume_dir, 'docker-compose.yaml')
    docker_status_path = os.path.join(volume_dir, 'status.json')
    if os.path.exists(docker_compose_config_path):
        os.remove(docker_compose_config_path)
    if os.path.exists(docker_status_path):
        os.remove(docker_status_path)
        with open(docker_status_path, 'wt') as f:
            f.write('{}')
    print(f'Starting monitor process. Please start Ray cluster with:\n   RAY_FAKE_CLUSTER=1 ray up {config_file}')
    monitor_docker(docker_compose_config_path, docker_status_path, project_name)