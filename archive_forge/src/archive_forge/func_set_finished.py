from typing import Dict, List, Optional, Union, TYPE_CHECKING
from ray.util.annotations import DeveloperAPI
def set_finished(self):
    """Marks the search algorithm as finished."""
    self._finished = True