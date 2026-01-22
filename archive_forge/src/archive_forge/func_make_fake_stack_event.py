import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_stack_event(id, name, status='CREATE_COMPLETED', resource_name='id'):
    event_id = uuid.uuid4().hex
    self_url = '{endpoint}/stacks/{name}/{id}/resources/{name}/events/{event}'
    resource_url = '{endpoint}/stacks/{name}/{id}/resources/{name}'
    return {'resource_name': id if resource_name == 'id' else name, 'event_time': '2017-03-26T19:38:18', 'links': [{'href': self_url.format(endpoint=ORCHESTRATION_ENDPOINT, name=name, id=id, event=event_id), 'rel': 'self'}, {'href': resource_url.format(endpoint=ORCHESTRATION_ENDPOINT, name=name, id=id), 'rel': 'resource'}, {'href': '{endpoint}/stacks/{name}/{id}'.format(endpoint=ORCHESTRATION_ENDPOINT, name=name, id=id), 'rel': 'stack'}], 'logical_resource_id': name, 'resource_status': status, 'resource_status_reason': '', 'physical_resource_id': id, 'id': event_id}