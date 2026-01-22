from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
    assert not isinstance(self.comparison_value, list), "Comparison value for '$lte' operation must not be a list."
    return {'range': {self.field_name: {'lte': self.comparison_value}}}