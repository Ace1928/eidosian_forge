from contextlib import ExitStack
import matplotlib.pyplot as plt
import pytest
def pytest_itemcollected(item):
    mpl_marker = item.get_closest_marker('mpl_image_compare')
    if mpl_marker is None:
        return
    mpl_marker.kwargs.setdefault('tolerance', 0.5)
    for path in item.fspath.parts(reverse=True):
        if path.basename == 'cartopy':
            return
        elif path.basename == 'tests':
            subdir = item.fspath.relto(path)[:-len(item.fspath.ext)]
            mpl_marker.kwargs.setdefault('baseline_dir', f'baseline_images/{subdir}')
            break