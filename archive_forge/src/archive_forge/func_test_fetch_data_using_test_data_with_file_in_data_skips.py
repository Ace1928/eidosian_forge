import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_fetch_data_using_test_data_with_file_in_data_skips(tmp_project_with_test_file):
    project = tmp_project_with_test_file
    path = str(project / 'examples')
    data = project / 'examples' / 'data' / 'test_data.csv'
    data.write_text(REAL_FILE_CONTENT)
    fetch_data(name='pyct', path=path, use_test_data=True)
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / 'test_data.csv').read_text() == REAL_FILE_CONTENT