import os
import voila.app
import pytest
@pytest.fixture
def voila_config():
    return lambda app: None