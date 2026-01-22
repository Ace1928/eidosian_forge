import logging
import os
import time
import unittest
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_privsep import comm
from oslo_privsep import priv_context
@test_context_with_timeout.entrypoint
def sleep_with_t_context(long_timeout=0.04):
    time.sleep(long_timeout)
    return 42