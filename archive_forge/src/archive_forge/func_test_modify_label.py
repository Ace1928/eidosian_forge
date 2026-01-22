from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Label
@pytest.mark.django_db
def test_modify_label(run_module, admin_user, organization):
    label = Label.objects.create(name='test-label', organization=organization)
    result = run_module('label', dict(name='test-label', new_name='renamed-label', organization=organization.name), admin_user)
    assert not result.get('failed'), result.get('msg', result)
    assert result.get('changed', False)
    label.refresh_from_db()
    assert label.organization == organization
    assert label.name == 'renamed-label'