import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_json_unknown_version(self):
    m = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2)
    with self.assertRaises(ValueError) as exc:
        json.loads(json.dumps({'m': m.serialize(serializer=serializers.JsonSerializer()), 'v': bakery.LATEST_VERSION + 1}), cls=bakery.MacaroonJSONDecoder)
    self.assertEqual('unknown bakery version 4', exc.exception.args[0])