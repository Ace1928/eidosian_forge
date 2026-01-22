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
def override_docker_image(config_yaml, docker_image):
    docker_config = config_yaml.get('docker', {})
    docker_config['image'] = docker_image
    docker_config['container_name'] = 'ray_container'
    assert docker_config.get('head_image') is None, 'Cannot override head_image'
    assert docker_config.get('worker_image') is None, 'Cannot override worker_image'
    config_yaml['docker'] = docker_config