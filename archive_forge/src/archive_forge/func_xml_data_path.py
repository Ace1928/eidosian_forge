from pathlib import Path
import pytest
@pytest.fixture
def xml_data_path():
    return Path(__file__).parent.parent / 'data' / 'xml'