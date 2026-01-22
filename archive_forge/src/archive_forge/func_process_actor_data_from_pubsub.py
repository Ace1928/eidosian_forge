import asyncio
import logging
import os
import time
from collections import deque
import aiohttp.web
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray._private.gcs_pubsub import GcsAioActorSubscriber
from ray.core.generated import (
from ray.dashboard.datacenter import DataSource, DataOrganizer
from ray.dashboard.modules.actor import actor_consts
from ray.dashboard.optional_utils import rest_response
def process_actor_data_from_pubsub(actor_id, actor_table_data):
    actor_table_data = actor_table_data_to_dict(actor_table_data)
    if actor_table_data['state'] != 'DEPENDENCIES_UNREADY':
        actors = DataSource.actors[actor_id]
        for k in state_keys:
            if k in actor_table_data:
                actors[k] = actor_table_data[k]
        actor_table_data = actors
    actor_id = actor_table_data['actorId']
    node_id = actor_table_data['address']['rayletId']
    if actor_table_data['state'] == 'DEAD':
        self.dead_actors_queue.append(actor_id)
    DataSource.actors[actor_id] = actor_table_data
    if node_id != actor_consts.NIL_NODE_ID:
        node_actors = DataSource.node_actors.get(node_id, {})
        node_actors[actor_id] = actor_table_data
        DataSource.node_actors[node_id] = node_actors