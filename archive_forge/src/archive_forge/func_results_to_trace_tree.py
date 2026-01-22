import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
@staticmethod
def results_to_trace_tree(request: Dict[str, Any], response: Response, results: List[trace_tree.Result], time_elapsed: float) -> trace_tree.WBTraceTree:
    """Converts the request, response, and results into a trace tree.

        params:
            request: The request dictionary
            response: The response object
            results: A list of results object
            time_elapsed: The time elapsed in seconds
        returns:
            A wandb trace tree object.
        """
    start_time_ms = int(round(response['created'] * 1000))
    end_time_ms = start_time_ms + int(round(time_elapsed * 1000))
    span = trace_tree.Span(name=f'{response.get('model', 'openai')}_{response['object']}_{response.get('created')}', attributes=dict(response), start_time_ms=start_time_ms, end_time_ms=end_time_ms, span_kind=trace_tree.SpanKind.LLM, results=results)
    model_obj = {'request': request, 'response': response, '_kind': 'openai'}
    return trace_tree.WBTraceTree(root_span=span, model_dict=model_obj)