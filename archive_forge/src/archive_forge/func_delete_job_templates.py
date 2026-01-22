import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def delete_job_templates(app, client, jt):
    if is_api_v2(app):
        jt_id = get_resource_id(client.job_templates, jt)
        client.job_templates.delete(jt_id)
    else:
        jt_id = get_resource_id(client.jobs, jt)
        client.jobs.delete(jt_id)