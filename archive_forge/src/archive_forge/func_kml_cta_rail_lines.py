from pathlib import Path
import pytest
@pytest.fixture
def kml_cta_rail_lines(xml_data_path, datapath):
    return datapath(xml_data_path / 'cta_rail_lines.kml')