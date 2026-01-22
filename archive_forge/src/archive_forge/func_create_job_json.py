import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def create_job_json(client, app, **template):
    if is_api_v2(app):
        data = client.jobs.create(**template).to_dict()
    else:
        data = client.job_executions.create(**template).to_dict()
    return data