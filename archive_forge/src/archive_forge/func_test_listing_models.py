import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_listing_models(self) -> None:
    model_info_list_1 = hub.list_models(self.repo, model='mnist', tags=['vision'])
    model_info_list_2 = hub.list_models(self.repo, tags=['vision'])
    model_info_list_3 = hub.list_models(self.repo)
    self.assertGreater(len(model_info_list_1), 1)
    self.assertGreater(len(model_info_list_2), len(model_info_list_1))
    self.assertGreater(len(model_info_list_3), len(model_info_list_2))