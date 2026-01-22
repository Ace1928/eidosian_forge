from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_conflicting_name_and_id(run_module, admin_user):
    """In the event that 2 related items match our search criteria in this way:
    one item has an id that matches input
    one item has a name that matches input
    We should preference the id over the name.
    Otherwise, the universality of the controller_api lookup plugin is compromised.
    """
    org_by_id = Organization.objects.create(name='foo')
    slug = str(org_by_id.id)
    Organization.objects.create(name=slug)
    result = run_module('team', {'name': 'foo_team', 'description': 'fooin around', 'organization': slug}, admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    team = Team.objects.filter(name='foo_team').first()
    assert str(team.organization_id) == slug, 'Lookup by id should be preferenced over name in cases of conflict.'
    assert team.organization.name == 'foo'