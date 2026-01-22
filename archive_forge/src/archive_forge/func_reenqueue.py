import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def reenqueue(self, batch):
    tp = batch.tp
    self._batches[tp].appendleft(batch)
    self._pending_batches.remove(batch)
    batch.reset_drain()