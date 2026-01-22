import contextlib
import functools
import inspect
import os
from platform import uname
from pathlib import Path
import shutil
import string
import sys
import warnings
from packaging.version import parse as parse_version
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure
def check_figures_equal(*, extensions=('png', 'pdf', 'svg'), tol=0):
    """
    Decorator for test cases that generate and compare two figures.

    The decorated function must take two keyword arguments, *fig_test*
    and *fig_ref*, and draw the test and reference images on them.
    After the function returns, the figures are saved and compared.

    This decorator should be preferred over `image_comparison` when possible in
    order to keep the size of the test suite from ballooning.

    Parameters
    ----------
    extensions : list, default: ["png", "pdf", "svg"]
        The extensions to test.
    tol : float
        The RMS threshold above which the test is considered failed.

    Raises
    ------
    RuntimeError
        If any new figures are created (and not subsequently closed) inside
        the test function.

    Examples
    --------
    Check that calling `.Axes.plot` with a single argument plots it against
    ``[0, 1, 2, ...]``::

        @check_figures_equal()
        def test_plot(fig_test, fig_ref):
            fig_test.subplots().plot([1, 3, 5])
            fig_ref.subplots().plot([0, 1, 2], [1, 3, 5])

    """
    ALLOWED_CHARS = set(string.digits + string.ascii_letters + '_-[]()')
    KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY

    def decorator(func):
        import pytest
        _, result_dir = _image_directories(func)
        old_sig = inspect.signature(func)
        if not {'fig_test', 'fig_ref'}.issubset(old_sig.parameters):
            raise ValueError(f"The decorated function must have at least the parameters 'fig_test' and 'fig_ref', but your function has the signature {old_sig}")

        @pytest.mark.parametrize('ext', extensions)
        def wrapper(*args, ext, request, **kwargs):
            if 'ext' in old_sig.parameters:
                kwargs['ext'] = ext
            if 'request' in old_sig.parameters:
                kwargs['request'] = request
            file_name = ''.join((c for c in request.node.name if c in ALLOWED_CHARS))
            try:
                fig_test = plt.figure('test')
                fig_ref = plt.figure('reference')
                with _collect_new_figures() as figs:
                    func(*args, fig_test=fig_test, fig_ref=fig_ref, **kwargs)
                if figs:
                    raise RuntimeError('Number of open figures changed during test. Make sure you are plotting to fig_test or fig_ref, or if this is deliberate explicitly close the new figure(s) inside the test.')
                test_image_path = result_dir / (file_name + '.' + ext)
                ref_image_path = result_dir / (file_name + '-expected.' + ext)
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                _raise_on_image_difference(ref_image_path, test_image_path, tol=tol)
            finally:
                plt.close(fig_test)
                plt.close(fig_ref)
        parameters = [param for param in old_sig.parameters.values() if param.name not in {'fig_test', 'fig_ref'}]
        if 'ext' not in old_sig.parameters:
            parameters += [inspect.Parameter('ext', KEYWORD_ONLY)]
        if 'request' not in old_sig.parameters:
            parameters += [inspect.Parameter('request', KEYWORD_ONLY)]
        new_sig = old_sig.replace(parameters=parameters)
        wrapper.__signature__ = new_sig
        new_marks = getattr(func, 'pytestmark', []) + wrapper.pytestmark
        wrapper.pytestmark = new_marks
        return wrapper
    return decorator