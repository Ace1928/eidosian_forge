from .metrics import HTTP_REQUEST_DURATION_SECONDS  # type:ignore[unused-ignore]

    Tornado log handler for recording RED metrics.

    We record the following metrics:
       Rate - the number of requests, per second, your services are serving.
       Errors - the number of failed requests per second.
       Duration - The amount of time each request takes expressed as a time interval.

    We use a fully qualified name of the handler as a label,
    rather than every url path to reduce cardinality.

    This function should be either the value of or called from a function
    that is the 'log_function' tornado setting. This makes it get called
    at the end of every request, allowing us to record the metrics we need.
    