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
def create_certs(tmpdir):
    """Create CURVE certificates for a test"""
    base_dir = str(tmpdir.join('certs').mkdir())
    keys_dir = os.path.join(base_dir, 'certificates')
    public_keys_dir = os.path.join(base_dir, 'public_keys')
    secret_keys_dir = os.path.join(base_dir, 'private_keys')
    os.mkdir(keys_dir)
    os.mkdir(public_keys_dir)
    os.mkdir(secret_keys_dir)
    server_public_file, server_secret_file = zmq.auth.create_certificates(keys_dir, 'server')
    client_public_file, client_secret_file = zmq.auth.create_certificates(keys_dir, 'client')
    for key_file in os.listdir(keys_dir):
        if key_file.endswith('.key'):
            shutil.move(os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, '.'))
    for key_file in os.listdir(keys_dir):
        if key_file.endswith('.key_secret'):
            shutil.move(os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, '.'))
    return (public_keys_dir, secret_keys_dir)