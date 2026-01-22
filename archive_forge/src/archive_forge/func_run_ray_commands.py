import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
import boto3
import yaml
from google.cloud import storage
import ray
def run_ray_commands(cluster_config, retries, no_config_cache, num_expected_nodes=1):
    """
    Run the necessary Ray commands to start a cluster, verify Ray is running, and clean
    up the cluster.

    Args:
        cluster_config: The path of the cluster configuration file.
        retries: The number of retries for the verification step.
        no_config_cache: Whether to pass the --no-config-cache flag to the ray CLI
            commands.
    """
    print('======================================')
    cleanup_cluster(cluster_config)
    print('======================================')
    print('Starting new cluster...')
    cmd = ['ray', 'up', '-v', '-y']
    if no_config_cache:
        cmd.append('--no-config-cache')
    cmd.append(str(cluster_config))
    print(' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(f'stdout:\n{e.stdout.decode('utf-8')}')
        print(f'stderr:\n{e.stderr.decode('utf-8')}')
        raise e
    print('======================================')
    print('Verifying Ray is running...')
    success = False
    count = 0
    while count < retries:
        try:
            cmd = ['ray', 'exec', '-v', str(cluster_config), 'python -c \'import ray; ray.init("localhost:6379");' + f" assert len(ray.nodes()) >= {num_expected_nodes}'"]
            if no_config_cache:
                cmd.append('--no-config-cache')
            subprocess.run(cmd, check=True)
            success = True
            break
        except subprocess.CalledProcessError:
            count += 1
            print(f'Verification failed. Retry attempt {count} of {retries}...')
            time.sleep(60)
    if not success:
        print('======================================')
        print(f'Error: Verification failed after {retries} attempts. Cleaning up cluster before exiting...')
        cleanup_cluster(cluster_config)
        print('======================================')
        print('Exiting script.')
        sys.exit(1)
    print('======================================')
    print('Ray verification successful.')
    cleanup_cluster(cluster_config)
    print('======================================')
    print('Finished executing script successfully.')