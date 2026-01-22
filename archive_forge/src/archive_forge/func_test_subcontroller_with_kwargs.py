import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
def test_subcontroller_with_kwargs(self):
    r = self.app_.post('/sub', dict(foo=1))
    assert r.status_int == 200
    assert b'subindex' in r.body