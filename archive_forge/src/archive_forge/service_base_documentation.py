from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_server_pb2 as spb
Base service abstract class.

Derived classes for socket service interfaces classes should implement
abstract methods.
