import pytest
import tempfile
import os
from unittest.mock import patch
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.resources import resource_find, resource_add_path
def test_load_resource_not_found():
    Cache.remove(RESOURCE_CACHE)
    missing_file_name = 'missing_test_file.foo'
    find_missing_file = resource_find(missing_file_name)
    assert find_missing_file is None
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_file_path = os.path.join(temp_dir, missing_file_name)
        with open(missing_file_path, 'w'):
            pass
        find_missing_file_again = resource_find(missing_file_name)
        assert find_missing_file_again is None
        cached_filename = Cache.get(RESOURCE_CACHE, missing_file_name)
        assert cached_filename is None
        resource_add_path(temp_dir)
        found_file = resource_find(missing_file_name)
        assert missing_file_path == found_file
        assert missing_file_path == Cache.get(RESOURCE_CACHE, missing_file_name)