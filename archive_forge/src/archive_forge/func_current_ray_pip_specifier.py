import hashlib
import json
import logging
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from filelock import FileLock
import ray
from ray._private.runtime_env.conda_utils import (
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.validation import parse_and_validate_conda
from ray._private.utils import (
def current_ray_pip_specifier(logger: Optional[logging.Logger]=default_logger) -> Optional[str]:
    """The pip requirement specifier for the running version of Ray.

    Returns:
        A string which can be passed to `pip install` to install the
        currently running Ray version, or None if running on a version
        built from source locally (likely if you are developing Ray).

    Examples:
        Returns "https://s3-us-west-2.amazonaws.com/ray-wheels/[..].whl"
            if running a stable release, a nightly or a specific commit
    """
    if os.environ.get('RAY_CI_POST_WHEEL_TESTS'):
        return os.path.join(Path(ray.__file__).resolve().parents[2], '.whl', get_wheel_filename())
    elif ray.__commit__ == '{{RAY_COMMIT_SHA}}':
        if os.environ.get('RAY_RUNTIME_ENV_LOCAL_DEV_MODE') != '1':
            logger.warning('Current Ray version could not be detected, most likely because you have manually built Ray from source.  To use runtime_env in this case, set the environment variable RAY_RUNTIME_ENV_LOCAL_DEV_MODE=1.')
        return None
    elif 'dev' in ray.__version__:
        if _is_m1_mac():
            raise ValueError('Nightly wheels are not available for M1 Macs.')
        return get_master_wheel_url()
    elif _is_m1_mac():
        return f'ray=={ray.__version__}'
    else:
        return get_release_wheel_url()