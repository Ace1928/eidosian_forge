import importlib
import importlib.metadata
import os
import shlex
import sys
import textwrap
import types
from flask import Flask, Response, send_from_directory
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.handlers import (
from mlflow.utils.os import get_entry_points, is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION
@app.route(_add_static_prefix('/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval'))
def serve_get_metric_history_bulk_interval():
    return get_metric_history_bulk_interval_handler()