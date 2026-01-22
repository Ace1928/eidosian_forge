import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_fetch_data_using_test_data_and_force_with_file_in_data_over_writes(tmp_project_with_test_file):
    project = tmp_project_with_test_file
    path = str(project / 'examples')
    data = project / 'examples' / 'data' / 'test_data.csv'
    data.write_text(REAL_FILE_CONTENT)
    fetch_data(name='pyct', path=path, use_test_data=True, force=True)
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / 'test_data.csv').read_text() == TEST_FILE_CONTENT