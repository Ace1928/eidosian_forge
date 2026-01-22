import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_up(test_dir):
    selector = FileSelector(test_dir)
    selector._directory.value = os.path.join(test_dir, 'subdir1')
    selector._go.clicks = 1
    assert selector._cwd == os.path.join(test_dir, 'subdir1')
    selector._up.clicks = 1
    assert selector._cwd == test_dir