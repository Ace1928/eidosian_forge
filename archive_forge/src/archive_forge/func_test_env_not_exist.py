from os import environ
import shlex
import subprocess
import sys
import pytest
def test_env_not_exist():
    env = _patch_env(ENV_NAME, *KIVY_ENVS_TO_EXCLUDE)
    stdout = _kivy_subproces_import(env)
    assert EXPECTED_STR in stdout