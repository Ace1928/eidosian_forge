import os
from abc import ABC, abstractmethod
import torch.testing._internal.dist_utils

        Returns a partial string indicating the error we should receive when an
        RPC has timed out. Useful for use with assertRaisesRegex() to ensure we
        have the right errors during timeout.
        