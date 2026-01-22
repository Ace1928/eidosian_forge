import itertools
from collections import defaultdict
import param
from ..converter import HoloViewsConverter
from ..util import is_list_like, process_dynamic_args
def explorer(self, x=None, y=None, **kwds):
    """
        The `explorer` plot allows you to interactively explore your data.

        Reference: https://hvplot.holoviz.org/user_guide/Explorer.html

        Parameters
        ----------
        x : string, optional
            The coordinate variable along the x-axis
        y : string, optional
            The coordinate variable along the y-axis
        **kwds : optional
            Additional keywords arguments typically passed to hvplot's call.

        Returns
        -------
        The corresponding explorer type based on data, e.g. hvplot.ui.hvDataFrameExplorer.

        Examples
        --------

        .. code-block:

            import hvplot.pandas
            import pandas as pd

            df = pd.DataFrame(
                {
                    "actual": [100, 150, 125, 140, 145, 135, 123],
                    "forecast": [90, 160, 125, 150, 141, 141, 120],
                    "numerical": [1.1, 1.9, 3.2, 3.8, 4.3, 5.0, 5.5],
                    "date": pd.date_range("2022-01-03", "2022-01-09"),
                    "string": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                },
            )
            df.hvplot.explorer()
        """
    from ..ui import explorer as ui_explorer
    return ui_explorer(self._data, x=x, y=y, **kwds)