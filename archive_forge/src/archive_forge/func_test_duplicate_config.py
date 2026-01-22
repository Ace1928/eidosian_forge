from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_duplicate_config(collection_import, silence_warning):
    ControllerAPIModule = collection_import('plugins.module_utils.controller_api').ControllerAPIModule
    data = {'name': 'zigzoom', 'zig': 'zoom', 'controller_username': 'bob', 'controller_config_file': 'my_config'}
    with mock.patch.object(ControllerAPIModule, 'load_config') as mock_load:
        argument_spec = dict(name=dict(required=True), zig=dict(type='str'))
        ControllerAPIModule(argument_spec=argument_spec, direct_params=data)
        assert mock_load.mock_calls[-1] == mock.call('my_config')
    silence_warning.assert_called_once_with('The parameter(s) controller_username were provided at the same time as controller_config_file. Precedence may be unstable, we suggest either using config file or params.')