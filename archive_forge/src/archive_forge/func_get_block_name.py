from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.plot import AltairPlot, AltairPlotData, Plot
def get_block_name(self) -> str:
    return 'plot'