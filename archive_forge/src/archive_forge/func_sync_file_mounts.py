import logging
import os
import subprocess
import time
import traceback
from threading import Thread
import click
from ray._private.usage import usage_constants, usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.tags import (
def sync_file_mounts(self, sync_cmd, step_numbers=(0, 2)):
    previous_steps, total_steps = step_numbers
    nolog_paths = []
    if cli_logger.verbosity == 0:
        nolog_paths = ['~/ray_bootstrap_key.pem', '~/ray_bootstrap_config.yaml']

    def do_sync(remote_path, local_path, allow_non_existing_paths=False):
        if allow_non_existing_paths and (not os.path.exists(local_path)):
            cli_logger.print('sync: {} does not exist. Skipping.', local_path)
            return
        assert os.path.exists(local_path), local_path
        if os.path.isdir(local_path):
            if not local_path.endswith('/'):
                local_path += '/'
            if not remote_path.endswith('/'):
                remote_path += '/'
        with LogTimer(self.log_prefix + 'Synced {} to {}'.format(local_path, remote_path)):
            is_docker = self.docker_config and self.docker_config['container_name'] != ''
            if not is_docker:
                self.cmd_runner.run('mkdir -p {}'.format(os.path.dirname(remote_path)), run_env='host')
            sync_cmd(local_path, remote_path, docker_mount_if_possible=True)
            if remote_path not in nolog_paths:
                cli_logger.print('{} from {}', cf.bold(remote_path), cf.bold(local_path))
    with cli_logger.group('Processing file mounts', _numbered=('[]', previous_steps + 1, total_steps)):
        for remote_path, local_path in self.file_mounts.items():
            do_sync(remote_path, local_path)
        previous_steps += 1
    if self.cluster_synced_files:
        with cli_logger.group('Processing worker file mounts', _numbered=('[]', previous_steps + 1, total_steps)):
            cli_logger.print('synced files: {}', str(self.cluster_synced_files))
            for path in self.cluster_synced_files:
                do_sync(path, path, allow_non_existing_paths=True)
            previous_steps += 1
    else:
        cli_logger.print('No worker file mounts to sync', _numbered=('[]', previous_steps + 1, total_steps))