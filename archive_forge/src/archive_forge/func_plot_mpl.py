import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def plot_mpl(mpl_fig, resize=False, strip_style=False, verbose=False, show_link=False, link_text='Export to plot.ly', validate=True, output_type='file', include_plotlyjs=True, filename='temp-plot.html', auto_open=True, image=None, image_filename='plot_image', image_height=600, image_width=800):
    """
    Convert a matplotlib figure to a Plotly graph stored locally as HTML.

    For more information on converting matplotlib visualizations to plotly
    graphs, call help(plotly.tools.mpl_to_plotly)

    For more information on creating plotly charts locally as an HTML document
    or string, call help(plotly.offline.plot)

    mpl_fig -- a matplotlib figure object to convert to a plotly graph

    Keyword arguments:
    resize (default=False) -- allow plotly to choose the figure size.
    strip_style (default=False) -- allow plotly to choose style options.
    verbose (default=False) -- print message.
    show_link (default=False) -- display a link in the bottom-right corner of
        of the chart that will export the chart to Plotly Cloud or
        Plotly Enterprise
    link_text (default='Export to plot.ly') -- the text of export link
    validate (default=True) -- validate that all of the keys in the figure
        are valid? omit if your version of plotly.js has become outdated
        with your version of graph_reference.json or if you need to include
        extra, unnecessary keys in your figure.
    output_type ('file' | 'div' - default 'file') -- if 'file', then
        the graph is saved as a standalone HTML file and `plot`
        returns None.
        If 'div', then `plot` returns a string that just contains the
        HTML <div> that contains the graph and the script to generate the
        graph.
        Use 'file' if you want to save and view a single graph at a time
        in a standalone HTML file.
        Use 'div' if you are embedding these graphs in an HTML file with
        other graphs or HTML markup, like a HTML report or an website.
    include_plotlyjs (default=True) -- If True, include the plotly.js
        source code in the output file or string.
        Set as False if your HTML file already contains a copy of the plotly.js
        library.
    filename (default='temp-plot.html') -- The local filename to save the
        outputted chart to. If the filename already exists, it will be
        overwritten. This argument only applies if `output_type` is 'file'.
    auto_open (default=True) -- If True, open the saved file in a
        web browser after saving.
        This argument only applies if `output_type` is 'file'.
    image (default=None |'png' |'jpeg' |'svg' |'webp') -- This parameter sets
        the format of the image to be downloaded, if we choose to download an
        image. This parameter has a default value of None indicating that no
        image should be downloaded.
    image_filename (default='plot_image') -- Sets the name of the file your
        image will be saved to. The extension should not be included.
    image_height (default=600) -- Specifies the height of the image in `px`.
    image_width (default=800) -- Specifies the width of the image in `px`.

    Example:
    ```
    from plotly.offline import init_notebook_mode, plot_mpl
    import matplotlib.pyplot as plt

    init_notebook_mode()

    fig = plt.figure()
    x = [10, 15, 20, 25, 30]
    y = [100, 250, 200, 150, 300]
    plt.plot(x, y, "o")

    plot_mpl(fig)
    # If you want to to download an image of the figure as well
    plot_mpl(fig, image='png')
    ```
    """
    plotly_plot = tools.mpl_to_plotly(mpl_fig, resize, strip_style, verbose)
    return plot(plotly_plot, show_link, link_text, validate, output_type, include_plotlyjs, filename, auto_open, image=image, image_filename=image_filename, image_height=image_height, image_width=image_width)