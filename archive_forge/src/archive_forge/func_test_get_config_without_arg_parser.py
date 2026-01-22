import argparse
import os_client_config
from os_client_config.tests import base
def test_get_config_without_arg_parser(self):
    cloud_config = os_client_config.get_config(options=None, validate=False)
    self.assertIsInstance(cloud_config, os_client_config.cloud_config.CloudConfig)