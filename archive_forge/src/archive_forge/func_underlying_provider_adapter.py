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
@property
def underlying_provider_adapter(self) -> ProviderAdapter:
    provider = self._underlying_provider
    if not provider:
        raise HTTPException(status_code=422, detail=f'Unknown Amazon Bedrock model type {self._underlying_provider}')
    adapter = provider.adapter
    if not adapter:
        raise HTTPException(status_code=422, detail=f"Don't know how to handle {self._underlying_provider} for Amazon Bedrock")
    return adapter