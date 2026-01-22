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
def check_arguments():
    """
    Check command line arguments and return the cluster configuration file path, the
    number of retries, the number of expected nodes, and the value of the
    --no-config-cache flag.
    """
    parser = argparse.ArgumentParser(description='Launch and verify a Ray cluster')
    parser.add_argument('--no-config-cache', action='store_true', help='Pass the --no-config-cache flag to Ray CLI commands')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for verifying Ray is running (default: 3)')
    parser.add_argument('--num-expected-nodes', type=int, default=1, help='Number of nodes for verifying Ray is running (default: 1)')
    parser.add_argument('--docker-override', choices=['disable', 'latest', 'nightly', 'commit'], default='disable', help='Override the docker image used for the head node and worker nodes')
    parser.add_argument('--wheel-override', type=str, default='', help='Override the wheel used for the head node and worker nodes')
    parser.add_argument('cluster_config', type=str, help='Path to the cluster configuration file')
    args = parser.parse_args()
    assert not (args.docker_override != 'disable' and args.wheel_override != ''), 'Cannot override both docker and wheel'
    return (args.cluster_config, args.retries, args.no_config_cache, args.num_expected_nodes, args.docker_override, args.wheel_override)