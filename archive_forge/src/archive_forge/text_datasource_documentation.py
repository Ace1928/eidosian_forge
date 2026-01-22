from typing import TYPE_CHECKING, Iterator, List
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
Text datasource, for reading and writing text files.