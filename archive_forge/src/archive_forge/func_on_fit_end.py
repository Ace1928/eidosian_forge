from collections import defaultdict
from copy import deepcopy
import torch
from typing import Any, Optional, Dict
import pytorch_lightning as pl  # type: ignore[import]
from ._data_sparstity_utils import (
def on_fit_end(self, trainer, pl_module) -> None:
    self.sparsified = deepcopy(pl_module.model).eval()
    self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
    _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)
    self.data_sparsifier.step()
    self.data_sparsifier.squash_mask()
    _log_sparsified_level(self.sparsified, self.data_sparsifier)