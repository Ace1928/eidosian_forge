import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_r_libnn(r_home: str):
    return _get_r_cmd_config(r_home, 'LIBnn', allow_empty=False)[0]