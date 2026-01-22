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
def override_wheels_url(config_yaml, wheel_url):
    setup_commands = config_yaml.get('setup_commands', [])
    setup_commands.append(f'pip3 uninstall -y ray && pip3 install -U "ray[default] @ {wheel_url}"')
    config_yaml['setup_commands'] = setup_commands