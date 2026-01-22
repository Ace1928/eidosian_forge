import glance_store
from unittest import mock
from glance.common import exception
from glance.common import store_utils
import glance.location
from glance.tests.unit import base
def test_add_location_with_restricted_sources(self):
    loc1 = {'url': 'file:///fake1.img.tar.gz', 'metadata': {}}
    loc2 = {'url': 'swift+config:///xxx', 'metadata': {}}
    loc3 = {'url': 'filesystem:///foo.img.tar.gz', 'metadata': {}}
    image1 = TestStoreLocation.FakeImageProxy()
    locations = glance.location.StoreLocations(image1, [])
    self.assertRaises(exception.BadStoreUri, locations.insert, 0, loc1)
    self.assertRaises(exception.BadStoreUri, locations.insert, 0, loc3)
    self.assertNotIn(loc1, locations)
    self.assertNotIn(loc3, locations)
    image2 = TestStoreLocation.FakeImageProxy()
    locations = glance.location.StoreLocations(image2, [loc1])
    self.assertRaises(exception.BadStoreUri, locations.insert, 0, loc2)
    self.assertNotIn(loc2, locations)