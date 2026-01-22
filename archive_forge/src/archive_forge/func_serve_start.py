import inspect
import logging
from types import FunctionType
from typing import Any, Dict, Tuple, Union
import ray
from ray._private.pydantic_compat import is_subclass_of_base_model
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.usage import usage_lib
from ray.actor import ActorHandle
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.context import _get_global_client, _set_global_client
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.schema import LoggingConfig
def serve_start(http_options: Union[None, dict, HTTPOptions]=None, grpc_options: Union[None, dict, gRPCOptions]=None, global_logging_config: Union[None, dict, LoggingConfig]=None, **kwargs) -> ServeControllerClient:
    """Initialize a serve instance.

    By default, the instance will be scoped to the lifetime of the returned
    Client object (or when the script exits). This is
    only relevant if connecting to a long-running Ray cluster (e.g., with
    ray.init(address="auto") or ray.init("ray://<remote_addr>")).

    Args:
        http_options (Optional[Dict, serve.HTTPOptions]): Configuration options
          for HTTP proxy. You can pass in a dictionary or HTTPOptions object
          with fields:

            - host(str, None): Host for HTTP servers to listen on. Defaults to
              "127.0.0.1". To expose Serve publicly, you probably want to set
              this to "0.0.0.0".
            - port(int): Port for HTTP server. Defaults to 8000.
            - root_path(str): Root path to mount the serve application
              (for example, "/serve"). All deployment routes will be prefixed
              with this path. Defaults to "".
            - middlewares(list): A list of Starlette middlewares that will be
              applied to the HTTP servers in the cluster. Defaults to [].
            - location(str, serve.config.DeploymentMode): The deployment
              location of HTTP servers:

                - "HeadOnly": start one HTTP server on the head node. Serve
                  assumes the head node is the node you executed serve.start
                  on. This is the default.
                - "EveryNode": start one HTTP server per node.
                - "NoServer" or None: disable HTTP server.
            - num_cpus (int): The number of CPU cores to reserve for each
              internal Serve HTTP proxy actor.  Defaults to 0.
        grpc_options: [Experimental] Configuration options for gRPC proxy.
          You can pass in a gRPCOptions object with fields:

            - port(int): Port for gRPC server. Defaults to 9000.
            - grpc_servicer_functions(list): List of import paths for gRPC
                `add_servicer_to_server` functions to add to Serve's gRPC proxy.
                Default empty list, meaning not to start the gRPC server.
    """
    usage_lib.record_library_usage('serve')
    try:
        client = _get_global_client(_health_check_controller=True)
        logger.info(f'Connecting to existing Serve app in namespace "{SERVE_NAMESPACE}". New http options will not be applied.')
        if http_options:
            _check_http_options(client, http_options)
        return client
    except RayServeException:
        pass
    controller, controller_name = _start_controller(http_options, grpc_options, global_logging_config, **kwargs)
    client = ServeControllerClient(controller, controller_name)
    _set_global_client(client)
    logger.info(f'Started Serve in namespace "{SERVE_NAMESPACE}".')
    return client