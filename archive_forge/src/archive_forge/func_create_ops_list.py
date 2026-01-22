from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
@property
def create_ops_list(self):
    """Returns create_ops_list function matching ``use_csingle`` precision."""
    return create_ops_listC64 if self.use_csingle else create_ops_listC128