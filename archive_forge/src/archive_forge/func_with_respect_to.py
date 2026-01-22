from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
@with_respect_to.setter
def with_respect_to(self, value):
    json_path = self._get_path('with_respect_to')
    value = self.panel_metrics_helper.front_to_back(value)
    nested_set(self, json_path, value)