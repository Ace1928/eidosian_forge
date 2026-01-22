import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def get_default_host_creds(store_uri):
    creds = read_mlflow_creds()
    return MlflowHostCreds(host=store_uri, username=creds.username, password=creds.password, token=MLFLOW_TRACKING_TOKEN.get(), aws_sigv4=MLFLOW_TRACKING_AWS_SIGV4.get(), auth=MLFLOW_TRACKING_AUTH.get(), ignore_tls_verification=MLFLOW_TRACKING_INSECURE_TLS.get(), client_cert_path=MLFLOW_TRACKING_CLIENT_CERT_PATH.get(), server_cert_path=MLFLOW_TRACKING_SERVER_CERT_PATH.get())