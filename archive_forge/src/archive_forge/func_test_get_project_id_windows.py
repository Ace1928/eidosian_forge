import io
import json
import os
import subprocess
import sys
import mock
import pytest  # type: ignore
from google.auth import _cloud_sdk
from google.auth import environment_vars
from google.auth import exceptions
@mock.patch('os.name', new='nt')
def test_get_project_id_windows():
    check_output_patch = mock.patch('subprocess.check_output', autospec=True, return_value=b'example-project\n')
    with check_output_patch as check_output:
        project_id = _cloud_sdk.get_project_id()
    assert project_id == 'example-project'
    assert check_output.called
    args = check_output.call_args[0]
    command = args[0]
    executable = command[0]
    assert executable == 'gcloud.cmd'