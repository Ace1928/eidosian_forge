import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_deal_with_pecan_configs(self):
    """If config comes from pecan.conf convert it to dict"""
    conf = Config(conf_dict={'items': ['url']})
    viewer = RequestViewerHook(conf)
    assert viewer.items == ['url']