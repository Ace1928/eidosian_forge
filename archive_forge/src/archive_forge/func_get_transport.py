from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_server_pb2 as spb
@abstractmethod
def get_transport(self) -> str:
    raise NotImplementedError