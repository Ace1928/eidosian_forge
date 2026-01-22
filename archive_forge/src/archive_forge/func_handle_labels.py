import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def handle_labels(labels):
    labels = format_labels(labels)
    if 'mesos_slave_executor_env_file' in labels:
        environment_variables_data = handle_json_from_file(labels['mesos_slave_executor_env_file'])
        labels['mesos_slave_executor_env_variables'] = jsonutils.dumps(environment_variables_data)
    return labels