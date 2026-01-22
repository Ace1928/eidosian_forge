import importlib.util
import os
import tempfile
from pathlib import PurePath
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union
import fsspec
import numpy as np
from .utils import logging
from .utils import tqdm as hf_tqdm
def load_elasticsearch_index(self, index_name: str, es_index_name: str, host: Optional[str]=None, port: Optional[int]=None, es_client: Optional['Elasticsearch']=None, es_index_config: Optional[dict]=None):
    """Load an existing text index using ElasticSearch for fast retrieval.

        Args:
            index_name (`str`):
                The `index_name`/identifier of the index. This is the index name that is used to call `get_nearest` or `search`.
            es_index_name (`str`):
                The name of elasticsearch index to load.
            host (`str`, *optional*, defaults to `localhost`):
                Host of where ElasticSearch is running.
            port (`str`, *optional*, defaults to `9200`):
                Port of where ElasticSearch is running.
            es_client (`elasticsearch.Elasticsearch`, *optional*):
                The elasticsearch client used to create the index if host and port are `None`.
            es_index_config (`dict`, *optional*):
                The configuration of the elasticsearch index.
                Default config is:
                    ```
                    {
                        "settings": {
                            "number_of_shards": 1,
                            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                        },
                        "mappings": {
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard",
                                    "similarity": "BM25"
                                },
                            }
                        },
                    }
                    ```
        """
    self._indexes[index_name] = ElasticSearchIndex(host=host, port=port, es_client=es_client, es_index_name=es_index_name, es_index_config=es_index_config)