import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_group(self, group, group_ref=None):
    self.assertIsNotNone(group.id)
    self.assertIn('self', group.links)
    self.assertIn('/groups/' + group.id, group.links['self'])
    if group_ref:
        self.assertEqual(group_ref['name'], group.name)
        self.assertEqual(group_ref['domain'], group.domain_id)
        if hasattr(group_ref, 'description'):
            self.assertEqual(group_ref['description'], group.description)
    else:
        self.assertIsNotNone(group.name)
        self.assertIsNotNone(group.domain_id)