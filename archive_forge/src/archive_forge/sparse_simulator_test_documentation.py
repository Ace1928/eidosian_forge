import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
A gate that modifies the target tensor in place, multiply by -1.