import argparse
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from testtools import matchers
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def new_deprecated():
    return cfg.DeprecatedOpt(uuid.uuid4().hex, group=uuid.uuid4().hex)