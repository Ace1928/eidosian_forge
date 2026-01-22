from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
def embedding_distance(self, prediction: str, reference: str, *, config: Optional[EmbeddingConfig]=None) -> _Matcher:
    """Compute the embedding distance between the prediction and reference.

        This logs the embedding distance to LangSmith and returns a `_Matcher` instance
        for making assertions on the distance value.

        By default, this uses the OpenAI API for computing embeddings.

        Args:
            prediction: The predicted string to compare.
            reference: The reference string to compare against.
            config: Optional configuration for the embedding distance evaluator.
                Supported options:
                - `encoder`: A custom encoder function to encode the list of input
                     strings to embeddings. Defaults to the OpenAI API.
                - `metric`: The distance metric to use for comparison.
                    Supported values: "cosine", "euclidean", "manhattan",
                    "chebyshev", "hamming".

        Returns:
            A `_Matcher` instance for the embedding distance value.


        Examples:
            >>> expect.embedding_distance(
            ...     prediction="hello",
            ...     reference="hi",
            ... ).to_be_less_than(1.0)
        """
    from langsmith._internal._embedding_distance import EmbeddingDistance
    config = config or {}
    encoder_func = 'custom' if config.get('encoder') else 'openai'
    evaluator = EmbeddingDistance(config=config)
    score = evaluator.evaluate(prediction=prediction, reference=reference)
    src_info = {'encoder': encoder_func, 'metric': evaluator.distance}
    self._submit_feedback('embedding_distance', {'score': score, 'source_info': src_info, 'comment': f'Using {encoder_func}, Metric: {evaluator.distance}'})
    return _Matcher(self.client, 'embedding_distance', score, _executor=self.executor)