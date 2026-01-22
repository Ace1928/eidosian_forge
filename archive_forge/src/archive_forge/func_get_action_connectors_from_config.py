import logging
from typing import Any, Tuple, TYPE_CHECKING
from ray.rllib.connectors.action.clip import ClipActionsConnector
from ray.rllib.connectors.action.immutable import ImmutableActionsConnector
from ray.rllib.connectors.action.lambdas import ConvertToNumpyConnector
from ray.rllib.connectors.action.normalize import NormalizeActionsConnector
from ray.rllib.connectors.action.pipeline import ActionConnectorPipeline
from ray.rllib.connectors.agent.clip_reward import ClipRewardAgentConnector
from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.connectors.agent.pipeline import AgentConnectorPipeline
from ray.rllib.connectors.agent.state_buffer import StateBufferConnector
from ray.rllib.connectors.agent.view_requirement import ViewRequirementAgentConnector
from ray.rllib.connectors.connector import Connector, ConnectorContext
from ray.rllib.connectors.registry import get_connector
from ray.rllib.connectors.agent.mean_std_filter import (
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.rllib.connectors.agent.synced_filter import SyncedFilterAgentConnector
@PublicAPI(stability='alpha')
def get_action_connectors_from_config(ctx: ConnectorContext, config: 'AlgorithmConfig') -> ActionConnectorPipeline:
    """Default list of action connectors to use for a new policy.

    Args:
        ctx: context used to create connectors.
        config: The AlgorithmConfig object.
    """
    connectors = [ConvertToNumpyConnector(ctx)]
    if config.get('normalize_actions', False):
        connectors.append(NormalizeActionsConnector(ctx))
    if config.get('clip_actions', False):
        connectors.append(ClipActionsConnector(ctx))
    connectors.append(ImmutableActionsConnector(ctx))
    return ActionConnectorPipeline(ctx, connectors)