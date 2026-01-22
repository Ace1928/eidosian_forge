import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def update_job_templates(app, client, jt_id, update_data):
    if is_api_v2(app):
        data = client.job_templates.update(jt_id, **update_data).job_template
    else:
        data = client.jobs.update(jt_id, **update_data).job
    return data