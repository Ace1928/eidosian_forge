from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def mock_stack(manager, stack_name, stack_id):
    return stacks.Stack(manager, {'id': stack_id, 'stack_name': stack_name, 'links': [{'href': 'http://192.0.2.1:8004/v1/1234/stacks/%s/%s' % (stack_name, stack_id), 'rel': 'self'}], 'description': 'No description', 'stack_status_reason': 'Stack create completed successfully', 'creation_time': '2013-08-04T20:57:55Z', 'updated_time': '2013-08-04T20:57:55Z', 'stack_status': 'CREATE_COMPLETE'})