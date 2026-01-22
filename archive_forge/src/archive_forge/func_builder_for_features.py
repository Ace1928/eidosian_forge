import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from . import (
def builder_for_features(self, *feature_list):
    cls = type('Builder_' + '_'.join(feature_list), (object,), {'features': feature_list})
    self.registry.register(cls)
    return cls