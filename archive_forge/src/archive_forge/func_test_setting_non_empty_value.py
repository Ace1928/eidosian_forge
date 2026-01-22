import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_setting_non_empty_value(self):
    uploader = FileUpload()
    content = memoryview(b'some content')
    uploader.value = [{'name': 'some-name.txt', 'type': 'text/plain', 'size': 561, 'last_modified': dt.datetime(2020, 1, 9, 13, 58, 16, 434000, tzinfo=dt.timezone.utc), 'content': content}]
    state = uploader.get_state(key='value')
    assert len(state['value']) == 1
    [entry] = state['value']
    assert entry['name'] == 'some-name.txt'
    assert entry['type'] == 'text/plain'
    assert entry['size'] == 561
    assert entry['last_modified'] == 1578578296434
    assert entry['content'] == content