import datetime
import os
def get_autodoc_datetime():
    """Obtain the datetime to use for timestamps embedded in generated docs.

    :return: A `datetime` object
    """
    try:
        return datetime.datetime.utcfromtimestamp(int(os.environ['SOURCE_DATE_EPOCH']))
    except (KeyError, ValueError):
        return datetime.datetime.utcnow()