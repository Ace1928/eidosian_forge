import os
import pytest
@pytest.fixture
def nameset():
    name = 'hey_i_am_an_env_var'
    os.environ[name] = 'i am a value'
    yield name
    del os.environ[name]