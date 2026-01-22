import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
@out_features.setter
def out_features(self, out_features: List[str]):
    """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
    self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features=out_features, out_indices=None, stage_names=self.stage_names)