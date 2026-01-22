import datetime
from unittest import mock
from testtools import matchers
from heat.engine.clients.os import swift
from heat.tests import common
from heat.tests import utils
def post_account(data):
    head_account.update(data)