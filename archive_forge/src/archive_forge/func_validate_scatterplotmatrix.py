from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def validate_scatterplotmatrix(df, index, diag, colormap_type, **kwargs):
    """
    Validates basic inputs for FigureFactory.create_scatterplotmatrix()

    :raises: (PlotlyError) If pandas is not imported
    :raises: (PlotlyError) If pandas dataframe is not inputted
    :raises: (PlotlyError) If pandas dataframe has <= 1 columns
    :raises: (PlotlyError) If diagonal plot choice (diag) is not one of
        the viable options
    :raises: (PlotlyError) If colormap_type is not a valid choice
    :raises: (PlotlyError) If kwargs contains 'size', 'color' or
        'colorscale'
    """
    if not pd:
        raise ImportError('FigureFactory.scatterplotmatrix requires a pandas DataFrame.')
    if not isinstance(df, pd.core.frame.DataFrame):
        raise exceptions.PlotlyError('Dataframe not inputed. Please use a pandas dataframe to produce a scatterplot matrix.')
    if len(df.columns) <= 1:
        raise exceptions.PlotlyError('Dataframe has only one column. To use the scatterplot matrix, use at least 2 columns.')
    if diag not in DIAG_CHOICES:
        raise exceptions.PlotlyError('Make sure diag is set to one of {}'.format(DIAG_CHOICES))
    if colormap_type not in VALID_COLORMAP_TYPES:
        raise exceptions.PlotlyError("Must choose a valid colormap type. Either 'cat' or 'seq' for a categorical and sequential colormap respectively.")
    if 'marker' in kwargs:
        FORBIDDEN_PARAMS = ['size', 'color', 'colorscale']
        if any((param in kwargs['marker'] for param in FORBIDDEN_PARAMS)):
            raise exceptions.PlotlyError("Your kwargs dictionary cannot include the 'size', 'color' or 'colorscale' key words inside the marker dict since 'size' is already an argument of the scatterplot matrix function and both 'color' and 'colorscale are set internally.")