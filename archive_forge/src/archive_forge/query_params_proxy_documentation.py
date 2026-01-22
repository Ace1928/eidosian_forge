from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping, overload
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state.query_params import missing_key_error_message
from streamlit.runtime.state.session_state_proxy import get_session_state

        Get all query parameters as a dictionary.

        This method primarily exists for internal use and is not needed for
        most cases. ``st.query_params`` returns an object that inherits from
        ``dict`` by default.

        When a key is repeated as a query parameter within the URL, this method
        will return only the last value of each unique key.

        Returns
        -------
        Dict[str,str]
            A dictionary of the current query paramters in the app's URL.
        