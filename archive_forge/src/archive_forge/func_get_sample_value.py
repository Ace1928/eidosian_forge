from abc import ABC, abstractmethod
import copy
from threading import Lock
from typing import Dict, Iterable, List, Optional
from .metrics_core import Metric
def get_sample_value(self, name: str, labels: Optional[Dict[str, str]]=None) -> Optional[float]:
    """Returns the sample value, or None if not found.

        This is inefficient, and intended only for use in unittests.
        """
    if labels is None:
        labels = {}
    for metric in self.collect():
        for s in metric.samples:
            if s.name == name and s.labels == labels:
                return s.value
    return None