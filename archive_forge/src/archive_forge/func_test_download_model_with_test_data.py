import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_download_model_with_test_data(self) -> None:
    directory = hub.download_model_with_test_data('mnist')
    files = os.listdir(directory)
    self.assertIsInstance(directory, str)
    self.assertIn(member='model.onnx', container=files, msg='Onnx model not found')
    self.assertIn(member='test_data_set_0', container=files, msg='Test data not found')