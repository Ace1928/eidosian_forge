import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def iplot(figure_or_data, show_link=False, link_text='Export to plot.ly', validate=True, image=None, filename='plot_image', image_width=800, image_height=600, config=None, auto_play=True, animation_opts=None):
    """
    Draw plotly graphs inside an IPython or Jupyter notebook

    figure_or_data -- a plotly.graph_objs.Figure or plotly.graph_objs.Data or
                      dict or list that describes a Plotly graph.
                      See https://plot.ly/python/ for examples of
                      graph descriptions.

    Keyword arguments:
    show_link (default=False) -- display a link in the bottom-right corner of
                                of the chart that will export the chart to
                                Plotly Cloud or Plotly Enterprise
    link_text (default='Export to plot.ly') -- the text of export link
    validate (default=True) -- validate that all of the keys in the figure
                               are valid? omit if your version of plotly.js
                               has become outdated with your version of
                               graph_reference.json or if you need to include
                               extra, unnecessary keys in your figure.
    image (default=None |'png' |'jpeg' |'svg' |'webp') -- This parameter sets
        the format of the image to be downloaded, if we choose to download an
        image. This parameter has a default value of None indicating that no
        image should be downloaded. Please note: for higher resolution images
        and more export options, consider using plotly.io.write_image. See
        https://plot.ly/python/static-image-export/ for more details.
    filename (default='plot') -- Sets the name of the file your image
        will be saved to. The extension should not be included.
    image_height (default=600) -- Specifies the height of the image in `px`.
    image_width (default=800) -- Specifies the width of the image in `px`.
    config (default=None) -- Plot view options dictionary. Keyword arguments
        `show_link` and `link_text` set the associated options in this
        dictionary if it doesn't contain them already.
    auto_play (default=True) -- Whether to automatically start the animation
        sequence on page load, if the figure contains frames. Has no effect if
        the figure does not contain frames.
    animation_opts (default=None) -- Dict of custom animation parameters that
        are used for the automatically started animation on page load. This
        dict is passed to the function Plotly.animate in Plotly.js. See
        https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js
        for available options. Has no effect if the figure
        does not contain frames, or auto_play is False.

    Example:
    ```
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode()
    iplot([{'x': [1, 2, 3], 'y': [5, 2, 7]}])
    # We can also download an image of the plot by setting the image to the
    format you want. e.g. `image='png'`
    iplot([{'x': [1, 2, 3], 'y': [5, 2, 7]}], image='png')
    ```

    animation_opts Example:
    ```
    from plotly.offline import iplot
    figure = {'data': [{'x': [0, 1], 'y': [0, 1]}],
              'layout': {'xaxis': {'range': [0, 5], 'autorange': False},
                         'yaxis': {'range': [0, 5], 'autorange': False},
                         'title': 'Start Title'},
              'frames': [{'data': [{'x': [1, 2], 'y': [1, 2]}]},
                         {'data': [{'x': [1, 4], 'y': [1, 4]}]},
                         {'data': [{'x': [3, 4], 'y': [3, 4]}],
                          'layout': {'title': 'End Title'}}]}
    iplot(figure, animation_opts={'frame': {'duration': 1}})
    ```
    """
    import plotly.io as pio
    ipython = get_module('IPython')
    if not ipython:
        raise ImportError('`iplot` can only run inside an IPython Notebook.')
    config = dict(config) if config else {}
    config.setdefault('showLink', show_link)
    config.setdefault('linkText', link_text)
    figure = tools.return_figure_from_figure_or_data(figure_or_data, validate)
    post_script = build_save_image_post_script(image, filename, image_height, image_width, 'iplot')
    pio.show(figure, validate=validate, config=config, auto_play=auto_play, post_script=post_script, animation_opts=animation_opts)