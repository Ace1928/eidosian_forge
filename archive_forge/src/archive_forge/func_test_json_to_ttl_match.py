import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import (
import os
from glob import glob
import logging
from prov.tests import examples
import prov.model as pm
import rdflib as rl
from rdflib.compare import graph_diff
from io import BytesIO, StringIO
def test_json_to_ttl_match(self):
    json_files = sorted(glob(os.path.join(os.path.dirname(__file__), 'json', '*.json')))
    skip = list(range(352, 380))
    skip_match = [5, 6, 7, 8, 15, 27, 28, 29, 75, 76, 77, 78, 79, 80, 260, 261, 262, 263, 264, 306, 313, 315, 317, 322, 323, 324, 325, 330, 332, 344, 346, 382, 389, 395, 397]
    errors = []
    for idx, fname in enumerate(json_files):
        _, ttl_file = os.path.split(fname)
        ttl_file = os.path.join(os.path.dirname(__file__), 'rdf', ttl_file.replace('json', 'ttl'))
        try:
            g = pm.ProvDocument.deserialize(fname)
            if len(g.bundles) == 0:
                format = 'turtle'
            else:
                format = 'trig'
            if format == 'trig':
                ttl_file = ttl_file.replace('ttl', 'trig')
            with open(ttl_file, 'rb') as fp:
                g_rdf = rl.ConjunctiveGraph().parse(fp, format=format)
            g0_rdf = rl.ConjunctiveGraph().parse(StringIO(g.serialize(format='rdf', rdf_format=format)), format=format)
            if idx not in skip_match:
                match, _, in_first, in_second = find_diff(g_rdf, g0_rdf)
                self.assertTrue(match)
            else:
                logger.info('Skipping match: %s' % fname)
            if idx in skip:
                logger.info('Skipping deserialization: %s' % fname)
                continue
            g1 = pm.ProvDocument.deserialize(content=g.serialize(format='rdf', rdf_format=format), format='rdf', rdf_format=format)
        except Exception as e:
            errors.append((e, idx, fname, in_first, in_second))
    self.assertFalse(errors)