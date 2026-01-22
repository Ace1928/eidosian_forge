import plotly
import plotly.graph_objs as go
from plotly.offline import get_plotlyjs_version
def validate_coerce_output_type(output_type):
    if output_type == 'Figure' or output_type == go.Figure:
        cls = go.Figure
    elif output_type == 'FigureWidget' or (hasattr(go, 'FigureWidget') and output_type == go.FigureWidget):
        cls = go.FigureWidget
    else:
        raise ValueError("\nInvalid output type: {output_type}\n    Must be one of: 'Figure', 'FigureWidget'")
    return cls