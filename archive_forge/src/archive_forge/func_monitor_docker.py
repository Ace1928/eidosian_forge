import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import yaml
def monitor_docker(docker_compose_path: str, status_path: str, project_name: str, update_interval: float=1.0):
    while not os.path.exists(docker_compose_path):
        time.sleep(0.5)
    print('Docker compose config detected, starting status monitoring')
    os.chmod(docker_compose_path, 511)
    docker_config = {'force_update': True}
    next_update = time.monotonic() - 1.0
    shutdown = False
    status = None
    while not shutdown:
        new_docker_config = _read_yaml(docker_compose_path)
        if new_docker_config != docker_config:
            shutdown = _update_docker_compose(docker_compose_path, project_name, status)
            next_update = time.monotonic() - 1.0
        if time.monotonic() > next_update:
            status = _update_docker_status(docker_compose_path, project_name, status_path)
            next_update = time.monotonic() + update_interval
        docker_config = new_docker_config
        time.sleep(0.1)
    print('Cluster shut down, terminating monitoring script.')