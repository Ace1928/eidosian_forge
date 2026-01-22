from typing import Dict, List
import catalogue
import pytest
from pytest import raises
from confection import Config, SimpleFrozenDict, SimpleFrozenList, registry
Test whether setting default values for a FrozenDict/FrozenList works within a config, which utilizes
    deepcopy.