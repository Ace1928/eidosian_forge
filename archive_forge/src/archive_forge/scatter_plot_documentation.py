from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal
from gradio_client.documentation import document
from gradio.components.plot import AltairPlot, AltairPlotData, Plot

        Parameters:
            value: Expects a pandas DataFrame containing the data to display in the scatter plot. The DataFrame should contain at least two columns, one for the x-axis (corresponding to this component's `x` argument) and one for the y-axis (corresponding to `y`).
        Returns:
            The data to display in a scatter plot, in the form of an AltairPlotData dataclass, which includes the plot information as a JSON string, as well as the type of plot (in this case, "scatter").
        