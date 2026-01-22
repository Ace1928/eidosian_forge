import logging
import threading
from typing import Dict, Optional
from wandb.proto.wandb_internal_pb2 import Record, Result
def context_id_from_record(record: Record) -> str:
    context_id = record.control.mailbox_slot
    return context_id