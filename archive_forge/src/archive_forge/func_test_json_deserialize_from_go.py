import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_json_deserialize_from_go(self):
    ns = checkers.Namespace()
    ns.register('someuri', 'x')
    m = bakery.Macaroon(root_key=b'rootkey', id=b'some id', location='here', version=bakery.LATEST_VERSION, namespace=ns)
    m.add_caveat(checkers.Caveat(condition='something', namespace='someuri'))
    data = '{"m":{"c":[{"i":"x:something"}],"l":"here","i":"some id","s64":"c8edRIupArSrY-WZfa62pgZFD8VjDgqho9U2PlADe-E"},"v":3,"ns":"someuri:x"}'
    m_go = bakery.Macaroon.deserialize_json(data)
    self.assertEqual(m.macaroon.signature_bytes, m_go.macaroon.signature_bytes)
    self.assertEqual(m.macaroon.version, m_go.macaroon.version)
    self.assertEqual(len(m_go.macaroon.caveats), 1)
    self.assertEqual(m.namespace, m_go.namespace)