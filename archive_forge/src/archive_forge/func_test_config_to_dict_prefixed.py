import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_to_dict_prefixed(self):
    from pecan import configuration
    'Add a prefix for keys'
    conf = configuration.initconf()
    assert isinstance(conf, configuration.Config)
    to_dict = conf.to_dict('prefix_')
    assert isinstance(to_dict, dict)
    assert to_dict['prefix_server']['prefix_host'] == '0.0.0.0'
    assert to_dict['prefix_server']['prefix_port'] == '8080'
    assert to_dict['prefix_app']['prefix_modules'] == []
    assert to_dict['prefix_app']['prefix_root'] is None
    assert to_dict['prefix_app']['prefix_static_root'] == 'public'
    assert to_dict['prefix_app']['prefix_template_path'] == ''