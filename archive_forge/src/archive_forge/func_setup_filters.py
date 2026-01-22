from __future__ import annotations
import warnings
from . import assertions
from .. import exc
from .. import exc as sa_exc
from ..exc import SATestSuiteWarning
from ..util.langhelpers import _warnings_warn
def setup_filters():
    """hook for setting up warnings filters.

    SQLAlchemy-specific classes must only be here and not in pytest config,
    as we need to delay importing SQLAlchemy until conftest.py has been
    processed.

    NOTE: filters on subclasses of DeprecationWarning or
    PendingDeprecationWarning have no effect if added here, since pytest
    will add at each test the following filters
    ``always::PendingDeprecationWarning`` and ``always::DeprecationWarning``
    that will take precedence over any added here.

    """
    warnings.filterwarnings('error', category=exc.SAWarning)
    warnings.filterwarnings('always', category=exc.SATestSuiteWarning)