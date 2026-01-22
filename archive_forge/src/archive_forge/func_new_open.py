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
def new_open(self, method, url, **kwargs):
    r = new_request(self, method, url, **kwargs)
    m = mock.MagicMock(read=mock.MagicMock(return_value=r._content), status=r.status_code, getheader=mock.MagicMock(side_effect=r.headers.get))
    return m