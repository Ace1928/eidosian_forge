import pytest
import tempfile
import os
from unittest.mock import patch
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.resources import resource_find, resource_add_path
def test_load_resource_cached(test_file):
    Cache.remove(RESOURCE_CACHE)
    found_file = resource_find(test_file)
    assert found_file is not None
    cached_filename = Cache.get(RESOURCE_CACHE, test_file)
    assert found_file == cached_filename