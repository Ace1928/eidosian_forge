from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from pluggy import HookspecMarker
@hookspec(firstresult=True)
def pytest_report_to_serializable(config: 'Config', report: Union['CollectReport', 'TestReport']) -> Optional[Dict[str, Any]]:
    """Serialize the given report object into a data structure suitable for
    sending over the wire, e.g. converted to JSON.

    :param config: The pytest config object.
    :param report: The report.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. The exact details may depend
    on the plugin which calls the hook.
    """