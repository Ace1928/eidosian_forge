import functools
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar
import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri
def get_databricks_host_creds(server_uri=None):
    """
    Reads in configuration necessary to make HTTP requests to a Databricks server. This
    uses the Databricks CLI's ConfigProvider interface to load the DatabricksConfig object.
    If no Databricks CLI profile is found corresponding to the server URI, this function
    will attempt to retrieve these credentials from the Databricks Secret Manager. For that to work,
    the server URI will need to be of the following format: "databricks://scope:prefix". In the
    Databricks Secret Manager, we will query for a secret in the scope "<scope>" for secrets with
    keys of the form "<prefix>-host" and "<prefix>-token". Note that this prefix *cannot* be empty
    if trying to authenticate with this method. If found, those host credentials will be used. This
    method will throw an exception if sufficient auth cannot be found.

    Args:
        server_uri: A URI that specifies the Databricks profile you want to use for making
            requests.

    Returns:
        MlflowHostCreds which includes the hostname and authentication information necessary to
        talk to the Databricks server.
    """
    profile, path = get_db_info_from_uri(server_uri)
    config = ProfileConfigProvider(profile).get_config() if profile else get_config()
    if (not config or not config.host) and path:
        dbutils = _get_dbutils()
        if dbutils:
            key_prefix = path
            host = dbutils.secrets.get(scope=profile, key=key_prefix + '-host')
            token = dbutils.secrets.get(scope=profile, key=key_prefix + '-token')
            if host and token:
                config = DatabricksConfig.from_token(host=host, token=token, insecure=False)
    if not config or not config.host:
        _fail_malformed_databricks_auth(profile)
    insecure = hasattr(config, 'insecure') and config.insecure
    if config.username is not None and config.password is not None:
        return MlflowHostCreds(config.host, username=config.username, password=config.password, ignore_tls_verification=insecure)
    elif config.token:
        return MlflowHostCreds(config.host, token=config.token, ignore_tls_verification=insecure)
    _fail_malformed_databricks_auth(profile)