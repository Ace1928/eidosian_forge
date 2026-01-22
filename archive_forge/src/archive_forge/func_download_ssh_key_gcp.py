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
def download_ssh_key_gcp():
    """Download the ssh key from the google cloud bucket to the local machine."""
    print('======================================')
    print('Downloading ssh key from GCP...')
    client = storage.Client()
    bucket_name = 'gcp-cluster-launcher-release-test-ssh-keys'
    key_name = 'ray-autoscaler_gcp_us-west1_anyscale-bridge-cd812d38_ubuntu_0.pem'
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(key_name)
    local_key_path = os.path.expanduser(f'~/.ssh/{key_name}')
    if not os.path.exists(os.path.dirname(local_key_path)):
        os.makedirs(os.path.dirname(local_key_path))
    blob.download_to_filename(local_key_path)
    os.chmod(local_key_path, 256)