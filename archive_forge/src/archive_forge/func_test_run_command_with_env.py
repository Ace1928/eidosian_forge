import re
import pytest
def test_run_command_with_env(protocol_real):
    shell_id = protocol_real.open_shell(env_vars=dict(TESTENV1='hi mom', TESTENV2='another var'))
    command_id = protocol_real.run_command(shell_id, 'echo', ['%TESTENV1%', '%TESTENV2%'])
    std_out, std_err, status_code = protocol_real.get_command_output(shell_id, command_id)
    assert re.search(b'hi mom another var', std_out)
    protocol_real.cleanup_command(shell_id, command_id)
    protocol_real.close_shell(shell_id)