from typing import Any, Dict, Optional
from uuid import uuid4
from tune.concepts.flow.report import TrialReport
def make_logger(obj: Any) -> 'MetricLogger':
    """Convert an object to a MetricLogger. This function is usually called on
    the worker side. If ``obj`` is a function, then it can take the context of
    the worker environment to initialize. For example mlflow will be able to take
    the worker side environment variables to initialize.

    :param obj: the object, currently we support ``MetricLogger`` or
        a callalbe generating a ``MetricLogger`` or ``None`` for a dummy logger
    :return: the logger
    """
    if obj is None:
        return MetricLogger()
    if isinstance(obj, MetricLogger):
        return obj
    if callable(obj):
        return obj()
    raise ValueError(f"{obj} can't be converted to a MetricLogger")