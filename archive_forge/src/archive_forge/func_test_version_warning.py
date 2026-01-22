from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_version_warning(collection_import, silence_warning):
    ControllerAPIModule = collection_import('plugins.module_utils.controller_api').ControllerAPIModule
    cli_data = {'ANSIBLE_MODULE_ARGS': {}}
    testargs = ['module_file2.py', json.dumps(cli_data)]
    with mock.patch.object(sys, 'argv', testargs):
        with mock.patch('ansible.module_utils.urls.Request.open', new=mock_awx_ping_response):
            my_module = ControllerAPIModule(argument_spec=dict())
            my_module._COLLECTION_VERSION = '2.0.0'
            my_module._COLLECTION_TYPE = 'awx'
            my_module.get_endpoint('ping')
    silence_warning.assert_called_once_with('You are running collection version {0} but connecting to {1} version {2}'.format(my_module._COLLECTION_VERSION, awx_name, ping_version))