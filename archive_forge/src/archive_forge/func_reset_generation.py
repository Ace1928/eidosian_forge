import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
def reset_generation(self):
    """ Coordinator did not recognize either generation or member_id. Will
        need to re-join the group.
        """
    self.generation = OffsetCommitRequest.DEFAULT_GENERATION_ID
    self.member_id = JoinGroupRequest[0].UNKNOWN_MEMBER_ID
    self.request_rejoin()