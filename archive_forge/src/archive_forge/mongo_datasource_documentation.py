import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
Datasource for reading from and writing to MongoDB.