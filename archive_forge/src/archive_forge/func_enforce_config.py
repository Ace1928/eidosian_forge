import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
@pytest.fixture(scope='session', autouse=True)
def enforce_config():
    """
    A fixture that ensures that all checks for MODIN_* variables
    are done using modin.config to prevent leakage
    """
    orig_env = os.environ
    modin_start = os.path.dirname(modin.__file__)
    modin_exclude = [os.path.dirname(modin.config.__file__), os.path.dirname(modin.tests.config.__file__)]

    class PatchedEnv:

        @staticmethod
        def __check_var(name):
            if name.upper().startswith('MODIN_'):
                frame = sys._getframe()
                try:
                    caller_file = frame.f_back.f_back.f_code.co_filename
                finally:
                    del frame
                pkg_name = os.path.dirname(caller_file)
                if pkg_name.startswith(modin_start):
                    assert any((pkg_name.startswith(excl) for excl in modin_exclude)), 'Do not access MODIN_ environment variable bypassing modin.config'

        def __getitem__(self, name):
            self.__check_var(name)
            return orig_env[name]

        def __setitem__(self, name, value):
            self.__check_var(name)
            orig_env[name] = value

        def __delitem__(self, name):
            self.__check_var(name)
            del orig_env[name]

        def pop(self, name, default=object()):
            self.__check_var(name)
            return orig_env.pop(name, default)

        def get(self, name, default=None):
            self.__check_var(name)
            return orig_env.get(name, default)

        def __contains__(self, name):
            self.__check_var(name)
            return name in orig_env

        def __getattr__(self, name):
            return getattr(orig_env, name)

        def __iter__(self):
            return iter(orig_env)
    os.environ = PatchedEnv()
    yield
    os.environ = orig_env