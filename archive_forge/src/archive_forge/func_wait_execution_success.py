import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def wait_execution_success(self, exec_id, timeout=180):
    start_time = time.time()
    ex = self.mistral_admin('execution-get', params=exec_id)
    exec_state = self.get_field_value(ex, 'State')
    expected_states = ['SUCCESS', 'RUNNING']
    while exec_state != 'SUCCESS':
        if time.time() - start_time > timeout:
            msg = 'Execution exceeds timeout {0} to change state to SUCCESS. Execution: {1}'.format(timeout, ex)
            raise exceptions.TimeoutException(msg)
        ex = self.mistral_admin('execution-get', params=exec_id)
        exec_state = self.get_field_value(ex, 'State')
        if exec_state not in expected_states:
            msg = 'Execution state %s is not in expected states: %s' % (exec_state, expected_states)
            raise exceptions.TempestException(msg)
        time.sleep(2)
    return True