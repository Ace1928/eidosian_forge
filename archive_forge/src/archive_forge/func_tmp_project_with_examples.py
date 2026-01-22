import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(scope='function')
def tmp_project_with_examples(tmp_path):
    project = tmp_path
    examples = project / 'examples'
    examples.mkdir()
    datasets = examples / 'datasets.yml'
    datasets.write_text(DATASETS_CONTENT)
    (examples / 'data').mkdir()
    example = examples / 'Test_Example_Notebook.ipynb'
    example.write_text(u'Fake notebook contents')
    return project