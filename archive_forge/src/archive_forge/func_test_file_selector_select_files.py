import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_select_files(test_dir):
    selector = FileSelector(test_dir)
    selector._directory.value = os.path.join(test_dir, 'subdir1')
    selector._go.clicks = 1
    selector._selector._lists[False].value = ['a']
    selector._selector._buttons[True].clicks = 1
    assert selector.value == [os.path.join(test_dir, 'subdir1', 'a')]
    selector._selector._lists[False].value = ['b']
    selector._selector._buttons[True].clicks = 2
    assert selector.value == [os.path.join(test_dir, 'subdir1', 'a'), os.path.join(test_dir, 'subdir1', 'b')]
    selector._selector._lists[True].value = ['a', 'b']
    selector._selector._buttons[False].clicks = 2
    assert selector.value == []