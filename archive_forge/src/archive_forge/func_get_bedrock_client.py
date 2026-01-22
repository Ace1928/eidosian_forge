import json
import time
from enum import Enum
import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import AmazonBedrockConfig, AWSIdAndKey, AWSRole, RouteConfig
from mlflow.gateway.constants import (
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys
from mlflow.gateway.schemas import completions
def get_bedrock_client(self):
    if self._client is not None and (not self._client_expired()):
        return self._client
    session = boto3.Session(**self._construct_session_args())
    try:
        self._client, self._client_created = (session.client(service_name='bedrock-runtime', **self._construct_client_args(session)), time.monotonic_ns())
        return self._client
    except botocore.exceptions.UnknownServiceError as e:
        raise AIGatewayConfigException('Cannot create Amazon Bedrock client; ensure boto3/botocore linked from the Amazon Bedrock user guide are installed. Otherwise likely missing credentials or accessing account without to Amazon Bedrock Private Preview') from e