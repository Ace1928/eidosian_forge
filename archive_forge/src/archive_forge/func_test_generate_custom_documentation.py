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
def test_generate_custom_documentation(self):
    reference = "startBlock{}\n  startItem{network}\n  endItem{network}\n  startBlock{network}\n    startItem{epanet file}\n      item{EPANET network inp file}\n    endItem{epanet file}\n  endBlock{network}\n  startItem{scenario}\n    item{Single scenario block}\n  endItem{scenario}\n  startBlock{scenario}\n    startItem{scenario file}\n      item{This is the (long) documentation for the 'scenario file'\nparameter.  It contains multiple lines, and some internal\nformatting; like a bulleted list:\n  - item 1\n  - item 2\n}\n    endItem{scenario file}\n    startItem{merlion}\n      item{This is the (long) documentation for the 'merlion' parameter.  It\n      contains multiple lines, but no apparent internal formatting; so the\n      outputter should re-wrap everything.}\n    endItem{merlion}\n    startItem{detection}\n      item{Sensor placement list, epanetID}\n    endItem{detection}\n  endBlock{scenario}\n  startItem{scenarios}\n    item{List of scenario blocks}\n  endItem{scenarios}\n  startBlock{scenarios}\n    startItem{scenario file}\n      item{This is the (long) documentation for the 'scenario file'\nparameter.  It contains multiple lines, and some internal\nformatting; like a bulleted list:\n  - item 1\n  - item 2\n}\n    endItem{scenario file}\n    startItem{merlion}\n      item{This is the (long) documentation for the 'merlion' parameter.  It\n      contains multiple lines, but no apparent internal formatting; so the\n      outputter should re-wrap everything.}\n    endItem{merlion}\n    startItem{detection}\n      item{Sensor placement list, epanetID}\n    endItem{detection}\n  endBlock{scenarios}\n  startItem{nodes}\n    item{List of node IDs}\n  endItem{nodes}\n  startItem{impact}\n  endItem{impact}\n  startBlock{impact}\n    startItem{metric}\n      item{Population or network based impact metric}\n    endItem{metric}\n  endBlock{impact}\n  startItem{flushing}\n  endItem{flushing}\n  startBlock{flushing}\n    startItem{flush nodes}\n    endItem{flush nodes}\n    startBlock{flush nodes}\n      startItem{feasible nodes}\n        item{ALL, NZD, NONE, list or filename}\n      endItem{feasible nodes}\n      startItem{infeasible nodes}\n        item{ALL, NZD, NONE, list or filename}\n      endItem{infeasible nodes}\n      startItem{max nodes}\n        item{Maximum number of nodes to flush}\n      endItem{max nodes}\n      startItem{rate}\n        item{Flushing rate [gallons/min]}\n      endItem{rate}\n      startItem{response time}\n        item{Time [min] between detection and flushing}\n      endItem{response time}\n      startItem{duration}\n        item{Time [min] for flushing}\n      endItem{duration}\n    endBlock{flush nodes}\n    startItem{close valves}\n    endItem{close valves}\n    startBlock{close valves}\n      startItem{feasible pipes}\n        item{ALL, DIAM min max [inch], NONE, list or filename}\n      endItem{feasible pipes}\n      startItem{infeasible pipes}\n        item{ALL, DIAM min max [inch], NONE, list or filename}\n      endItem{infeasible pipes}\n      startItem{max pipes}\n        item{Maximum number of pipes to close}\n      endItem{max pipes}\n      startItem{response time}\n        item{Time [min] between detection and closing valves}\n      endItem{response time}\n    endBlock{close valves}\n  endBlock{flushing}\nendBlock{}\n"
    with LoggingIntercept() as LOG:
        test = self.config.generate_documentation(block_start='startBlock{%s}\n', block_end='endBlock{%s}\n', item_start='startItem{%s}\n', item_body='item{%s}\n', item_end='endItem{%s}\n')
    LOG = LOG.getvalue().replace('\n', ' ')
    for name in ('block_start', 'block_end', 'item_start', 'item_end', 'item_body'):
        self.assertIn(f"Overriding '{name}' by passing strings to generate_documentation is deprecated.", LOG)
    self.maxDiff = None
    self.assertEqual(test, reference)
    with LoggingIntercept() as LOG:
        test = self.config.generate_documentation(format=String_ConfigFormatter(block_start='startBlock{%s}\n', block_end='endBlock{%s}\n', item_start='startItem{%s}\n', item_body='item{%s}\n', item_end='endItem{%s}\n'))
    self.assertEqual(LOG.getvalue(), '')
    self.maxDiff = None
    self.assertEqual(test, reference)
    with LoggingIntercept() as LOG:
        test = self.config.generate_documentation(block_start='startBlock\n', block_end='endBlock\n', item_start='startItem\n', item_body='item\n', item_end='endItem\n')
    stripped_reference = re.sub('\\{[^\\}]*\\}', '', reference, flags=re.M)
    self.assertEqual(test, stripped_reference)
    reference = 'startBlock{}\n  startBlock{network}\n  startBlock{scenario}\n  startBlock{scenarios}\n  startBlock{impact}\n  startBlock{flushing}\n    startBlock{flush nodes}\n    startBlock{close valves}\n'
    with LoggingIntercept() as LOG:
        test = self.config.generate_documentation(block_start='startBlock{%s}\n', block_end='', item_start='', item_body='')
    LOG = LOG.getvalue().replace('\n', ' ')
    for name in ('block_start', 'block_end', 'item_start', 'item_body'):
        self.assertIn(f"Overriding '{name}' by passing strings to generate_documentation is deprecated.", LOG)
    for name in 'item_end':
        self.assertNotIn(f"Overriding '{name}' by passing strings to generate_documentation is deprecated.", LOG)
    self.maxDiff = None
    self.assertEqual(test, reference)