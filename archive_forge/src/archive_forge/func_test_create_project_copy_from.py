from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Project
@pytest.mark.django_db
def test_create_project_copy_from(run_module, admin_user, organization, silence_warning):
    """Test the copy_from functionality"""
    result = run_module('project', dict(name='foo', organization=organization.name, scm_type='git', scm_url='https://foo.invalid', wait=False, scm_update_cache_timeout=5), admin_user)
    assert result.pop('changed', None), result
    proj_name = 'bar'
    result = run_module('project', dict(name=proj_name, copy_from='foo', scm_type='git', wait=False), admin_user)
    assert result.pop('changed', None), result
    result = run_module('project', dict(name=proj_name, copy_from='foo', scm_type='git', wait=False), admin_user)
    silence_warning.assert_called_with('A project with the name {0} already exists.'.format(proj_name))