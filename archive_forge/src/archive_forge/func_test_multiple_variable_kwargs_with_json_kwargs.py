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
def test_multiple_variable_kwargs_with_json_kwargs(self):
    r = self.app_.post_json('/variable_kwargs', {'id': '3', 'dummy': 'dummy'})
    assert r.status_int == 200
    assert r.body == b'variable_kwargs: dummy=dummy, id=3'