from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
@orientation.setter
def orientation(self, value):
    json_path = self._get_path('orientation')
    value = True if value == 'v' else False
    nested_set(self, json_path, value)