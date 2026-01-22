import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def assert_dict_is_subset(self, expected, actual):
    """Check if expected keys/values exist in actual response body.

        Check if the expected keys and values are in the actual response body.

        :param expected: dict of key-value pairs that are expected to be in
                         'actual' dict.
        :param actual: dict of key-value pairs.
        """
    for key, value in expected.items():
        self.assertEqual(value, actual[key])