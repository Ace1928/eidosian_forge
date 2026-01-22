import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
@pytest.fixture
def public_keys_dir(create_certs):
    public_keys_dir, secret_keys_dir = create_certs
    return public_keys_dir