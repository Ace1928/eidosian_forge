from typing import List
import datetime
import pytest
import numpy as np
import sympy
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor, VALID_LANGUAGES
Tests for SimulatedLocalProcessor