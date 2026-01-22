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
@pytest.fixture
def s3_storage_options(worker_id):
    if GithubCI.get():
        url = 'http://localhost:5000/'
    else:
        worker_id = '5' if worker_id == 'master' else worker_id.lstrip('gw')
        url = f'http://127.0.0.1:555{worker_id}/'
    return {'client_kwargs': {'endpoint_url': url}}