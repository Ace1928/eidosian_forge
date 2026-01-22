import flask_restful
import functools
import http.client
from oslo_log import log
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.server import flask as ks_flask
Revoke an inherited grant for a group on a project.

        DELETE /OS-INHERIT/projects/{project_id}/groups/{group_id}
               /roles/{role_id}/inherited_to_projects
        