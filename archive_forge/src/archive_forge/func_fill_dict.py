import heapq
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set
from triad import SerializableRLock
from triad.utils.convert import to_datetime
from tune._utils import to_base64
from tune.concepts.flow.trial import Trial
from tune.concepts.space.parameters import TuningParametersTemplate, to_template
from tune.constants import TUNE_REPORT, TUNE_REPORT_ID, TUNE_REPORT_METRIC
def fill_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill a row of :class:`~tune.concepts.dataset.StudyResult` with
        the report information

        :param data: a row (as dict) from :class:`~tune.concepts.dataset.StudyResult`
        :return: the updated ``data``
        """
    data[TUNE_REPORT_ID] = self.trial_id
    data[TUNE_REPORT_METRIC] = self.sort_metric
    data[TUNE_REPORT] = to_base64(self)
    return data