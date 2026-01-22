import functools
import importlib
import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional, Union
import sqlalchemy
from flask import Flask, Response, flash, jsonify, make_response, render_template_string, request
from werkzeug.datastructures import Authorization
from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import MANAGE, Permission, get_permission
from mlflow.server.auth.routes import (
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
from mlflow.store.entities import PagedList
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils
@catch_mlflow_exception
def get_registered_model_permission():
    name = _get_request_param('name')
    username = _get_request_param('username')
    rmp = store.get_registered_model_permission(name, username)
    return make_response({'registered_model_permission': rmp.to_json()})