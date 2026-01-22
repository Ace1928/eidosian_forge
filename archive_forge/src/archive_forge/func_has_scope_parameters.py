import abc
import json
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@property
def has_scope_parameters(self):
    """Return true if parameters can be used to create a scoped token."""
    return self.domain_id or self.domain_name or self.project_id or self.project_name or self.trust_id or self.system_scope