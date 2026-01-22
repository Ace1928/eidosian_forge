import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
@property
def out_feature_channels(self):
    return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}