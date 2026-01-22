import os
import sys
import json
import string
import shutil
import logging
import coloredlogs
import fire
import requests
from .._utils import run_command_with_process, compute_md5, job
@job('run `npm ci`')
def npm(self):
    """Job to install npm packages."""
    os.chdir(self.main)
    run_command_with_process('npm ci')