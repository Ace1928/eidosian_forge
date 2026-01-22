import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def make_v3_service(svc):
    eps = list(make_v3_endpoints(svc.endpoints))
    service = {'endpoints': eps, 'id': svc.id, 'type': svc.type}
    service['name'] = svc.extra.get('name', '')
    return service