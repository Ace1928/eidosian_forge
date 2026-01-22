import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_basic_usage(self) -> None:
    model = hub.load(self.name, self.repo)
    self.assertIsInstance(model, ModelProto)
    cached_files = list(glob.glob(join(hub.get_dir(), '**', '*.onnx'), recursive=True))
    self.assertGreaterEqual(len(cached_files), 1)