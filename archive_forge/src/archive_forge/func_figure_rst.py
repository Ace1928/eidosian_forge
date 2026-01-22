from glob import glob
import os
import shutil
import plotly
def figure_rst(figure_list, sources_dir):
    """Generate RST for a list of PNG filenames.

    Depending on whether we have one or more figures, we use a
    single rst call to 'image' or a horizontal list.

    Parameters
    ----------
    figure_list : list
        List of strings of the figures' absolute paths.
    sources_dir : str
        absolute path of Sphinx documentation sources

    Returns
    -------
    images_rst : str
        rst code to embed the images in the document
    """
    figure_paths = [os.path.relpath(figure_path, sources_dir).replace(os.sep, '/').lstrip('/') for figure_path in figure_list]
    images_rst = ''
    if not figure_paths:
        return images_rst
    figure_name = figure_paths[0]
    ext = os.path.splitext(figure_name)[1]
    figure_path = os.path.join('images', os.path.basename(figure_name))
    images_rst = SINGLE_HTML % figure_path
    return images_rst