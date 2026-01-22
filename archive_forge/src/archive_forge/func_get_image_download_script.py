import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def get_image_download_script(caller):
    """
    This function will return a script that will download an image of a Plotly
    plot.

    Keyword Arguments:
    caller ('plot', 'iplot') -- specifies which function made the call for the
        download script. If `iplot`, then an extra condition is added into the
        download script to ensure that download prompts aren't initiated on
        page reloads.
    """
    if caller == 'iplot':
        check_start = "if(document.readyState == 'complete') {{"
        check_end = '}}'
    elif caller == 'plot':
        check_start = ''
        check_end = ''
    else:
        raise ValueError('caller should only be one of `iplot` or `plot`')
    return "function downloadimage(format, height, width, filename) {{var p = document.getElementById('{{plot_id}}');Plotly.downloadImage(p, {{format: format, height: height, width: width, filename: filename}});}};" + check_start + "downloadimage('{format}', {height}, {width}, '{filename}');" + check_end