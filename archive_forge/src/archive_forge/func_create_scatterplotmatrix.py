from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def create_scatterplotmatrix(df, index=None, endpts=None, diag='scatter', height=500, width=500, size=6, title='Scatterplot Matrix', colormap=None, colormap_type='cat', dataframe=None, headers=None, index_vals=None, **kwargs):
    """
    Returns data for a scatterplot matrix;
    **deprecated**,
    use instead the plotly.graph_objects trace
    :class:`plotly.graph_objects.Splom`.

    :param (array) df: array of the data with column headers
    :param (str) index: name of the index column in data array
    :param (list|tuple) endpts: takes an increasing sequece of numbers
        that defines intervals on the real line. They are used to group
        the entries in an index of numbers into their corresponding
        interval and therefore can be treated as categorical data
    :param (str) diag: sets the chart type for the main diagonal plots.
        The options are 'scatter', 'histogram' and 'box'.
    :param (int|float) height: sets the height of the chart
    :param (int|float) width: sets the width of the chart
    :param (float) size: sets the marker size (in px)
    :param (str) title: the title label of the scatterplot matrix
    :param (str|tuple|list|dict) colormap: either a plotly scale name,
        an rgb or hex color, a color tuple, a list of colors or a
        dictionary. An rgb color is of the form 'rgb(x, y, z)' where
        x, y and z belong to the interval [0, 255] and a color tuple is a
        tuple of the form (a, b, c) where a, b and c belong to [0, 1].
        If colormap is a list, it must contain valid color types as its
        members.
        If colormap is a dictionary, all the string entries in
        the index column must be a key in colormap. In this case, the
        colormap_type is forced to 'cat' or categorical
    :param (str) colormap_type: determines how colormap is interpreted.
        Valid choices are 'seq' (sequential) and 'cat' (categorical). If
        'seq' is selected, only the first two colors in colormap will be
        considered (when colormap is a list) and the index values will be
        linearly interpolated between those two colors. This option is
        forced if all index values are numeric.
        If 'cat' is selected, a color from colormap will be assigned to
        each category from index, including the intervals if endpts is
        being used
    :param (dict) **kwargs: a dictionary of scatterplot arguments
        The only forbidden parameters are 'size', 'color' and
        'colorscale' in 'marker'

    Example 1: Vanilla Scatterplot Matrix

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create dataframe
    >>> df = pd.DataFrame(np.random.randn(10, 2),
    ...                 columns=['Column 1', 'Column 2'])

    >>> # Create scatterplot matrix
    >>> fig = create_scatterplotmatrix(df)
    >>> fig.show()


    Example 2: Indexing a Column

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create dataframe with index
    >>> df = pd.DataFrame(np.random.randn(10, 2),
    ...                    columns=['A', 'B'])

    >>> # Add another column of strings to the dataframe
    >>> df['Fruit'] = pd.Series(['apple', 'apple', 'grape', 'apple', 'apple',
    ...                          'grape', 'pear', 'pear', 'apple', 'pear'])

    >>> # Create scatterplot matrix
    >>> fig = create_scatterplotmatrix(df, index='Fruit', size=10)
    >>> fig.show()


    Example 3: Styling the Diagonal Subplots

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create dataframe with index
    >>> df = pd.DataFrame(np.random.randn(10, 4),
    ...                    columns=['A', 'B', 'C', 'D'])

    >>> # Add another column of strings to the dataframe
    >>> df['Fruit'] = pd.Series(['apple', 'apple', 'grape', 'apple', 'apple',
    ...                          'grape', 'pear', 'pear', 'apple', 'pear'])

    >>> # Create scatterplot matrix
    >>> fig = create_scatterplotmatrix(df, diag='box', index='Fruit', height=1000,
    ...                                width=1000)
    >>> fig.show()


    Example 4: Use a Theme to Style the Subplots

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create dataframe with random data
    >>> df = pd.DataFrame(np.random.randn(100, 3),
    ...                    columns=['A', 'B', 'C'])

    >>> # Create scatterplot matrix using a built-in
    >>> # Plotly palette scale and indexing column 'A'
    >>> fig = create_scatterplotmatrix(df, diag='histogram', index='A',
    ...                                colormap='Blues', height=800, width=800)
    >>> fig.show()


    Example 5: Example 4 with Interval Factoring

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd

    >>> # Create dataframe with random data
    >>> df = pd.DataFrame(np.random.randn(100, 3),
    ...                    columns=['A', 'B', 'C'])

    >>> # Create scatterplot matrix using a list of 2 rgb tuples
    >>> # and endpoints at -1, 0 and 1
    >>> fig = create_scatterplotmatrix(df, diag='histogram', index='A',
    ...                                colormap=['rgb(140, 255, 50)',
    ...                                          'rgb(170, 60, 115)', '#6c4774',
    ...                                          (0.5, 0.1, 0.8)],
    ...                                endpts=[-1, 0, 1], height=800, width=800)
    >>> fig.show()


    Example 6: Using the colormap as a Dictionary

    >>> from plotly.graph_objs import graph_objs
    >>> from plotly.figure_factory import create_scatterplotmatrix

    >>> import numpy as np
    >>> import pandas as pd
    >>> import random

    >>> # Create dataframe with random data
    >>> df = pd.DataFrame(np.random.randn(100, 3),
    ...                    columns=['Column A',
    ...                             'Column B',
    ...                             'Column C'])

    >>> # Add new color column to dataframe
    >>> new_column = []
    >>> strange_colors = ['turquoise', 'limegreen', 'goldenrod']

    >>> for j in range(100):
    ...     new_column.append(random.choice(strange_colors))
    >>> df['Colors'] = pd.Series(new_column, index=df.index)

    >>> # Create scatterplot matrix using a dictionary of hex color values
    >>> # which correspond to actual color names in 'Colors' column
    >>> fig = create_scatterplotmatrix(
    ...     df, diag='box', index='Colors',
    ...     colormap= dict(
    ...         turquoise = '#00F5FF',
    ...         limegreen = '#32CD32',
    ...         goldenrod = '#DAA520'
    ...     ),
    ...     colormap_type='cat',
    ...     height=800, width=800
    ... )
    >>> fig.show()
    """
    if dataframe is None:
        dataframe = []
    if headers is None:
        headers = []
    if index_vals is None:
        index_vals = []
    validate_scatterplotmatrix(df, index, diag, colormap_type, **kwargs)
    if isinstance(colormap, dict):
        colormap = clrs.validate_colors_dict(colormap, 'rgb')
    elif isinstance(colormap, str) and 'rgb' not in colormap and ('#' not in colormap):
        if colormap not in clrs.PLOTLY_SCALES.keys():
            raise exceptions.PlotlyError("If 'colormap' is a string, it must be the name of a Plotly Colorscale. The available colorscale names are {}".format(clrs.PLOTLY_SCALES.keys()))
        else:
            colormap = clrs.colorscale_to_colors(clrs.PLOTLY_SCALES[colormap])
            colormap = [colormap[0]] + [colormap[-1]]
        colormap = clrs.validate_colors(colormap, 'rgb')
    else:
        colormap = clrs.validate_colors(colormap, 'rgb')
    if not index:
        for name in df:
            headers.append(name)
        for name in headers:
            dataframe.append(df[name].values.tolist())
        utils.validate_dataframe(dataframe)
        figure = scatterplot(dataframe, headers, diag, size, height, width, title, **kwargs)
        return figure
    else:
        if index not in df:
            raise exceptions.PlotlyError('Make sure you set the index input variable to one of the column names of your dataframe.')
        index_vals = df[index].values.tolist()
        for name in df:
            if name != index:
                headers.append(name)
        for name in headers:
            dataframe.append(df[name].values.tolist())
        utils.validate_dataframe(dataframe)
        utils.validate_index(index_vals)
        if isinstance(colormap, dict):
            for key in colormap:
                if not all((index in colormap for index in index_vals)):
                    raise exceptions.PlotlyError('If colormap is a dictionary, all the names in the index must be keys.')
            figure = scatterplot_dict(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs)
            return figure
        else:
            figure = scatterplot_theme(dataframe, headers, diag, size, height, width, title, index, index_vals, endpts, colormap, colormap_type, **kwargs)
            return figure