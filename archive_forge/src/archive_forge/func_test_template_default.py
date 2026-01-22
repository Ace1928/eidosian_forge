import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_template_default(self):
    reference_template = '# Basic configuration for Flushing models\nnetwork:\n  epanet file: Net3.inp     # EPANET network inp file\nscenario:                   # Single scenario block\n  scenario file: Net3.tsg   # Scenario generation file, see the TEVASIM\n                            #   documentation\n  merlion: false            # Water quality model\n  detection: [1, 2, 3]      # Sensor placement list, epanetID\nscenarios: []               # List of scenario blocks\nnodes: []                   # List of node IDs\nimpact:\n  metric: MC                # Population or network based impact metric\nflushing:\n  flush nodes:\n    feasible nodes: ALL     # ALL, NZD, NONE, list or filename\n    infeasible nodes: NONE  # ALL, NZD, NONE, list or filename\n    max nodes: 2            # Maximum number of nodes to flush\n    rate: 600.0             # Flushing rate [gallons/min]\n    response time: 60.0     # Time [min] between detection and flushing\n    duration: 600.0         # Time [min] for flushing\n  close valves:\n    feasible pipes: ALL     # ALL, DIAM min max [inch], NONE, list or filename\n    infeasible pipes: NONE  # ALL, DIAM min max [inch], NONE, list or filename\n    max pipes: 2            # Maximum number of pipes to close\n    response time: 60.0     # Time [min] between detection and closing valves\n'
    self._validateTemplate(self.config, reference_template)