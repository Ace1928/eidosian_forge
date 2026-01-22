import re
import sys
from html import escape
from weakref import proxy
from .std import tqdm as std_tqdm
@staticmethod
def status_printer(_, total=None, desc=None, ncols=None):
    """
        Manage the printing of an IPython/Jupyter Notebook progress bar widget.
        """
    if IProgress is None:
        raise ImportError(WARN_NOIPYW)
    if total:
        pbar = IProgress(min=0, max=total)
    else:
        pbar = IProgress(min=0, max=1)
        pbar.value = 1
        pbar.bar_style = 'info'
        if ncols is None:
            pbar.layout.width = '20px'
    ltext = HTML()
    rtext = HTML()
    if desc:
        ltext.value = desc
    container = TqdmHBox(children=[ltext, pbar, rtext])
    if ncols is not None:
        ncols = str(ncols)
        try:
            if int(ncols) > 0:
                ncols += 'px'
        except ValueError:
            pass
        pbar.layout.flex = '2'
        container.layout.width = ncols
        container.layout.display = 'inline-flex'
        container.layout.flex_flow = 'row wrap'
    return container