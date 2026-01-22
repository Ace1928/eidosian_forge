from __future__ import annotations
import io
from abc import abstractmethod
from typing import NamedTuple, Protocol, Sequence
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider
Return a list of UploadFileUrlInfo for a given sequence of file_names.
        Optional to implement, issuing of URLs could be done by other service.

        Parameters
        ----------
        session_id
            The ID of the session that request URLs.
        file_names
            The sequence of file names for which URLs are requested

        Returns
        -------
        List[UploadFileUrlInfo]
            A list of UploadFileUrlInfo instances, each instance contains information
            about uploaded file URLs.
        