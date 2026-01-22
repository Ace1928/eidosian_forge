from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import os_service_types
from os_service_types.tests import base

test_builtin
------------

Tests for `ServiceTypes` class builtin data.

oslotest sets up a TempHomeDir for us, so there should be no ~/.cache files
available in these tests.
