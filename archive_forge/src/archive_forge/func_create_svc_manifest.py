from __future__ import absolute_import
import os
from os import path
import sys
import uuid
import logging
from kubernetes import client, config
from . import tracker
import yaml
def create_svc_manifest(name, port, target_port):
    spec = client.V1ServiceSpec(selector={'app': name}, ports=[client.V1ServicePort(protocol='TCP', port=port, target_port=target_port)])
    service = client.V1Service(metadata=client.V1ObjectMeta(name=name), spec=spec)
    return service