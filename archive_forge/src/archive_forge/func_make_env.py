import copy
import datetime
import io
import os
from oslo_serialization import jsonutils
import queue
import sys
import fixtures
import testtools
from magnumclient.common import httpclient as http
from magnumclient import shell
def make_env(self, exclude=None, fake_env=FAKE_ENV):
    env = dict(((k, v) for k, v in fake_env.items() if k != exclude))
    self.useFixture(fixtures.MonkeyPatch('os.environ', env))