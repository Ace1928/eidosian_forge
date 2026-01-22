import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
@property
def channels(self):
    return [self.out_feature_channels[name] for name in self.out_features]