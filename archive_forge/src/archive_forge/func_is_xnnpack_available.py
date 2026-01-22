import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack
        return str(torch.backends.xnnpack.enabled)
    else:
        return 'N/A'