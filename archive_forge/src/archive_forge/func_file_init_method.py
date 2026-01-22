import os
from abc import ABC, abstractmethod
import torch.testing._internal.dist_utils
@property
def file_init_method(self):
    return torch.testing._internal.dist_utils.INIT_METHOD_TEMPLATE.format(file_name=self.file_name)