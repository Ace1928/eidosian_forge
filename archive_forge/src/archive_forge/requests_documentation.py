from __future__ import absolute_import
import requests
from wandb_graphql.execution import ExecutionResult
from wandb_graphql.language.printer import print_ast
from .http import HTTPTransport

        :param url: The GraphQL URL
        :param auth: Auth tuple or callable to enable Basic/Digest/Custom HTTP Auth
        :param use_json: Send request body as JSON instead of form-urlencoded
        :param timeout: Specifies a default timeout for requests (Default: None)
        