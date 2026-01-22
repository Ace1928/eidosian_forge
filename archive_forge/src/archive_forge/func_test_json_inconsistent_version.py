import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_json_inconsistent_version(self):
    m = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V1)
    with self.assertRaises(ValueError) as exc:
        json.loads(json.dumps({'m': json.loads(m.serialize(serializer=serializers.JsonSerializer())), 'v': bakery.LATEST_VERSION}), cls=bakery.MacaroonJSONDecoder)
    self.assertEqual('underlying macaroon has inconsistent version; got 1 want 2', exc.exception.args[0])