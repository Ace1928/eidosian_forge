import ast
import sys
from functools import wraps
from typing import Callable, List, TypeVar
import requests
from wandb_gql.client import RetryError
from wandb import env
from wandb.errors import CommError, Error
Function decorator for catching common errors and re-raising as wandb.Error.