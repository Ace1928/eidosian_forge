import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
def is_version_available(version: str) -> bool:
    if 'git:' in version:
        return True
    is_available = True
    url = get_download_url(version)
    for i in range(6):
        try:
            urllib.request.urlopen(url)
        except urllib.error.HTTPError as err:
            print(f'Release {version} is unavailable from URL {url}')
            print(f'HTTPError: {err.code}')
            is_available = False
            break
        except urllib.error.URLError as e:
            if i < 5:
                print('checking version {} availability, retry ({}/5)'.format(version, i + 1))
                sleep(1)
                continue
            print('Release {} is unavailable from URL {}'.format(version, url))
            print('URLError: {}'.format(e.reason))
            is_available = False
    return is_available