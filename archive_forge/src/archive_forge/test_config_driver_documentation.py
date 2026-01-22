import tempfile
from oslo_config import cfg
from oslo_config import fixture
from oslotest import base
from castellan import _config_driver
from castellan.common.objects import opaque_data
from castellan.tests.unit.key_manager import fake

Functional test cases for the Castellan Oslo Config Driver.

Note: This requires local running instance of Vault.
