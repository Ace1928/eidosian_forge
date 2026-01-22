import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
 Add a memory policy that returns a false pointer 64 bytes into the
    actual allocation, and fill the prefix with some text. Then check at each
    memory manipulation that the prefix exists, to make sure all alloc/realloc/
    free/calloc go via the functions here.
    