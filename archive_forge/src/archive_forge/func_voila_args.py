import os
from lxml import etree
import pytest
from conftest import BASE_DIR
@pytest.fixture
def voila_args():
    nb_path = os.path.join(BASE_DIR, 'nb_report.ipynb')
    return [nb_path, '--VoilaTest.config_file_paths=[]']