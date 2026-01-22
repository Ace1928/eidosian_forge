import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import setuptools
def is_hip():
    import torch
    return torch.version.hip is not None