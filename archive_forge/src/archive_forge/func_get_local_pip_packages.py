import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
def get_local_pip_packages(archive: Archive):
    """Get currently installed pip packages and write into an archive.

    Args:
        archive: Archive object to add meta files to.

    Returns:
        Open archive object.
    """
    if not archive.is_open:
        archive.open()
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze
    with tempfile.NamedTemporaryFile('wt') as fp:
        for line in freeze.freeze():
            fp.writelines([line, '\n'])
        fp.flush()
        with archive.subdir('') as sd:
            sd.add(fp.name, 'pip_packages.txt')
    return archive