from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
import io
import json
from testtools import ExpectedException
from vitrageclient.common.formatters import DOTFormatter
from vitrageclient.common.formatters import GraphMLFormatter
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.topology import TopologyShow
def test_dot_emitter(self):

    def dict2columns(data):
        return zip(*sorted(data.items()))
    out = io.StringIO()
    formatter = DOTFormatter()
    topology = json.loads(JSON_DATA)
    columns, topology = dict2columns(topology)
    formatter.emit_one(columns, topology, out)
    self.assertEqual(DOT_DATA, out.getvalue())