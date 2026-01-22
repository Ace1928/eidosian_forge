import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
def parse_cluster_info(address: Optional[str]=None, create_cluster_if_needed: bool=False, cookies: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None) -> ClusterInfo:
    """Create a cluster if needed and return its address, cookies, and metadata."""
    if address is None:
        if ray.is_initialized() and ray._private.worker.global_worker.node.address_info['webui_url'] is not None:
            address = f'http://{ray._private.worker.global_worker.node.address_info['webui_url']}'
            logger.info(f'No address provided but Ray is running; using address {address}.')
        else:
            logger.info(f'No address provided, defaulting to {DEFAULT_DASHBOARD_ADDRESS}.')
            address = DEFAULT_DASHBOARD_ADDRESS
    if address == 'auto':
        raise ValueError("Internal error: unexpected address 'auto'.")
    if '://' not in address:
        logger.info(f"No scheme (e.g. 'http://') or module string (e.g. 'ray://') provided in address {address}, defaulting to HTTP.")
        address = f'http://{address}'
    module_string, inner_address = split_address(address)
    if module_string == 'ray':
        raise ValueError(f'Internal error: unexpected Ray Client address {address}.')
    if module_string in {'http', 'https'}:
        return get_job_submission_client_cluster_info(inner_address, create_cluster_if_needed=create_cluster_if_needed, cookies=cookies, metadata=metadata, headers=headers, _use_tls=module_string == 'https')
    else:
        try:
            module = importlib.import_module(module_string)
        except Exception:
            raise RuntimeError(f'Module: {module_string} does not exist.\nThis module was parsed from address: {address}') from None
        assert 'get_job_submission_client_cluster_info' in dir(module), f'Module: {module_string} does not have `get_job_submission_client_cluster_info`.\nThis module was parsed from address: {address}'
        return module.get_job_submission_client_cluster_info(inner_address, create_cluster_if_needed=create_cluster_if_needed, cookies=cookies, metadata=metadata, headers=headers)