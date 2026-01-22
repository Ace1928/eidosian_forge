from __future__ import absolute_import, division, print_function
import io
import os
import json
import datetime
import importlib
from contextlib import redirect_stdout, suppress
from unittest import mock
import logging
from requests.models import Response, PreparedRequest
import pytest
from ansible.module_utils.six import raise_from
from awx.main.tests.functional.conftest import _request
from awx.main.tests.functional.conftest import credentialtype_scm, credentialtype_ssh  # noqa: F401; pylint: disable=unused-variable
from awx.main.models import (
from django.db import transaction
@pytest.fixture
def collection_import():
    """These tests run assuming that the awx_collection folder is inserted
    into the PATH before-hand by collection_path_set.
    But all imports internally to the collection
    go through this fixture so that can be changed if needed.
    For instance, we could switch to fully-qualified import paths.
    """

    def rf(path):
        return importlib.import_module(path)
    return rf