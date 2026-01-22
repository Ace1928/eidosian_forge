import json
import os
import ssl
import sys
import warnings
import logging
import random
import testtools
import unittest
from unittest import mock
from os_ken.base import app_manager  # To suppress cyclic import
from os_ken.controller import controller
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_0_parser
def test_ports_accessibility_v10(self):
    self._test_ports_accessibility(ofproto_v1_0_parser, 0)