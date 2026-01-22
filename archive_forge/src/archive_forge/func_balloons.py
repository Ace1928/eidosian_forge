from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Balloons_pb2 import Balloons as BalloonsProto
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('balloons')
def balloons(self) -> DeltaGenerator:
    """Draw celebratory balloons.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.balloons()

        ...then watch your app and get ready for a celebration!

        """
    balloons_proto = BalloonsProto()
    balloons_proto.show = True
    return self.dg._enqueue('balloons', balloons_proto)