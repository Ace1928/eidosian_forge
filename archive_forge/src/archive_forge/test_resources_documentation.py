import pytest
import tempfile
import os
from unittest.mock import patch
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.resources import resource_find, resource_add_path

Resource loading tests
======================
