import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_copy_examples_using_force_with_prexisting_content_in_target(tmp_project_with_examples):
    project = tmp_project_with_examples
    path = str(project / 'examples')
    copy_examples(name='pyct', path=path, force=True)
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').is_file()
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').read_text() == EXAMPLE_CONTENT