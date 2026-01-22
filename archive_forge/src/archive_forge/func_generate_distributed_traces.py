from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
def generate_distributed_traces(self, otel_service_name, ansible_playbook, tasks_data, status, traceparent, disable_logs, disable_attributes_in_logs):
    """ generate distributed traces from the collected TaskData and HostData """
    tasks = []
    parent_start_time = None
    for task_uuid, task in tasks_data.items():
        if parent_start_time is None:
            parent_start_time = task.start
        tasks.append(task)
    trace.set_tracer_provider(TracerProvider(resource=Resource.create({SERVICE_NAME: otel_service_name})))
    processor = BatchSpanProcessor(OTLPSpanExporter())
    trace.get_tracer_provider().add_span_processor(processor)
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(ansible_playbook, context=self.traceparent_context(traceparent), start_time=parent_start_time, kind=SpanKind.SERVER) as parent:
        parent.set_status(status)
        if self.ansible_version is not None:
            parent.set_attribute('ansible.version', self.ansible_version)
        parent.set_attribute('ansible.session', self.session)
        parent.set_attribute('ansible.host.name', self.host)
        if self.ip_address is not None:
            parent.set_attribute('ansible.host.ip', self.ip_address)
        parent.set_attribute('ansible.host.user', self.user)
        for task in tasks:
            for host_uuid, host_data in task.host_data.items():
                with tracer.start_as_current_span(task.name, start_time=task.start, end_on_exit=False) as span:
                    self.update_span_data(task, host_data, span, disable_logs, disable_attributes_in_logs)