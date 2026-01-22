import http.client as http
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_resource_type import ResourceType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociations
from glance.api.v2.model.metadef_resource_type import ResourceTypes
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
ResourceTypeAssociation resource factory method