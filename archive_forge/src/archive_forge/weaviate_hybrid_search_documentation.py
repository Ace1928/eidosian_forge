from __future__ import annotations
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
Look up similar documents in Weaviate.

        query: The query to search for relevant documents
         of using weviate hybrid search.

        where_filter: A filter to apply to the query.
            https://weaviate.io/developers/weaviate/guides/querying/#filtering

        score: Whether to include the score, and score explanation
            in the returned Documents meta_data.

        hybrid_search_kwargs: Used to pass additional arguments
         to the .with_hybrid() method.
            The primary uses cases for this are:
            1)  Search specific properties only -
                specify which properties to be used during hybrid search portion.
                Note: this is not the same as the (self.attributes) to be returned.
                Example - hybrid_search_kwargs={"properties": ["question", "answer"]}
            https://weaviate.io/developers/weaviate/search/hybrid#selected-properties-only

            2) Weight boosted searched properties -
                Boost the weight of certain properties during the hybrid search portion.
                Example - hybrid_search_kwargs={"properties": ["question^2", "answer"]}
            https://weaviate.io/developers/weaviate/search/hybrid#weight-boost-searched-properties

            3) Search with a custom vector - Define a different vector
                to be used during the hybrid search portion.
                Example - hybrid_search_kwargs={"vector": [0.1, 0.2, 0.3, ...]}
            https://weaviate.io/developers/weaviate/search/hybrid#with-a-custom-vector

            4) Use Fusion ranking method
                Example - from weaviate.gql.get import HybridFusion
                hybrid_search_kwargs={"fusion": fusion_type=HybridFusion.RELATIVE_SCORE}
            https://weaviate.io/developers/weaviate/search/hybrid#fusion-ranking-method
        