import json
import logging
import os
import urllib
from typing import TYPE_CHECKING, Any, Dict, Optional
import requests
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.launch.utils import LAUNCH_DEFAULT_PROJECT
from wandb.sdk.lib import retry, runid
from wandb.sdk.lib.gql_request import GraphQLSession
def create_run_queue(self, name: str, type: 'public.RunQueueResourceType', entity: Optional[str]=None, prioritization_mode: Optional['public.RunQueuePrioritizationMode']=None, config: Optional[dict]=None, template_variables: Optional[dict]=None) -> 'public.RunQueue':
    """Create a new run queue (launch).

        Arguments:
            name: (str) Name of the queue to create
            type: (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex".
            entity: (str) Optional name of the entity to create the queue. If None, will use the configured or default entity.
            prioritization_mode: (str) Optional version of prioritization to use. Either "V0" or None
            config: (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. "{{var}}") to specify template variables.
            template_variables (dict): A dictionary of template variable schemas to be used with the config. Expected format of:
                {
                    "var-name": {
                        "schema": {
                            "type": "<string | number | integer>",
                            "default": <optional value>,
                            "minimum": <optional minimum>,
                            "maximum": <optional maximum>,
                            "enum": [..."<options>"]
                        }
                    }
                }

        Returns:
            The newly created `RunQueue`

        Raises:
            ValueError if any of the parameters are invalid
            wandb.Error on wandb API errors
        """
    if entity is None:
        entity = self.settings['entity'] or self.default_entity
        if entity is None:
            raise ValueError('entity must be passed as a parameter, or set in settings')
    if len(name) == 0:
        raise ValueError('name must be non-empty')
    if len(name) > 64:
        raise ValueError('name must be less than 64 characters')
    if type not in ['local-container', 'local-process', 'kubernetes', 'sagemaker', 'gcp-vertex']:
        raise ValueError("resource_type must be one of 'local-container', 'local-process', 'kubernetes', 'sagemaker', or 'gcp-vertex'")
    if prioritization_mode:
        prioritization_mode = prioritization_mode.upper()
        if prioritization_mode not in ['V0']:
            raise ValueError("prioritization_mode must be 'V0' if present")
    if config is None:
        config = {}
    self.create_project(LAUNCH_DEFAULT_PROJECT, entity)
    api = InternalApi(default_settings={'entity': entity, 'project': self.project(LAUNCH_DEFAULT_PROJECT)}, retry_timedelta=RETRY_TIMEDELTA)
    config_json = json.dumps({'resource_args': {type: config}})
    create_config_result = api.create_default_resource_config(entity, type, config_json, template_variables)
    if not create_config_result['success']:
        raise wandb.Error('failed to create default resource config')
    config_id = create_config_result['defaultResourceConfigID']
    create_queue_result = api.create_run_queue(entity, LAUNCH_DEFAULT_PROJECT, name, 'PROJECT', prioritization_mode, config_id)
    if not create_queue_result['success']:
        raise wandb.Error('failed to create run queue')
    return public.RunQueue(client=self.client, name=name, entity=entity, prioritization_mode=prioritization_mode, _access='PROJECT', _default_resource_config_id=config_id, _default_resource_config=config)