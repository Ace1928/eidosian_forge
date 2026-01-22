import os
import pathlib
import re
import sys
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import click
import watchfiles
import yaml
import ray
from ray import serve
from ray._private.utils import import_attr
from ray.autoscaler._private.cli_logger import cli_logger
from ray.dashboard.modules.dashboard_sdk import parse_runtime_env_args
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve._private import api as _private_api
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve.config import DeploymentMode, ProxyLocation, gRPCOptions
from ray.serve.deployment import Application, deployment_to_schema
from ray.serve.schema import (
def build_app_config(import_path: str, name: str=None):
    app: Application = import_attr(import_path)
    if not isinstance(app, Application):
        raise TypeError(f"Expected '{import_path}' to be an Application but got {type(app)}.")
    deployments = pipeline_build(app, name)
    ingress = get_and_validate_ingress_deployment(deployments)
    schema = ServeApplicationSchema(name=name, route_prefix=ingress.route_prefix, import_path=import_path, runtime_env={}, deployments=[deployment_to_schema(d, include_route_prefix=False) for d in deployments])
    return schema.dict(exclude_unset=True)