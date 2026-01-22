import logging
import os
from oslo_config import cfg
from oslo_middleware import cors
from oslo_policy import opts
from oslo_policy import policy
from paste import deploy
from glance.i18n import _
from glance.version import version_info as version
def set_cors_middleware_defaults():
    """Update default configuration options for oslo.middleware."""
    cors.set_defaults(allow_headers=['Content-MD5', 'X-Image-Meta-Checksum', 'X-Storage-Token', 'Accept-Encoding', 'X-Auth-Token', 'X-Identity-Status', 'X-Roles', 'X-Service-Catalog', 'X-User-Id', 'X-Tenant-Id', 'X-OpenStack-Request-ID'], expose_headers=['X-Image-Meta-Checksum', 'X-Auth-Token', 'X-Subject-Token', 'X-Service-Token', 'X-OpenStack-Request-ID'], allow_methods=['GET', 'PUT', 'POST', 'DELETE', 'PATCH'])