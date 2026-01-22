import os
from unittest import mock
import fixtures
import oslo_config
from oslotest import base as test_base
from oslo_policy import _cache_handler as _ch
Test the cache handler module