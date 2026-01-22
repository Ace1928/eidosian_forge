import logging
import os
import numpy as np
import pytest
@pytest.fixture(scope='session')
def save_figs(request):
    """Enable command line switch for saving generation figures upon testing."""
    fig_dir = request.config.getoption('--save')
    if fig_dir is not None:
        _log.info('Saving generated images in %s', fig_dir)
        os.makedirs(fig_dir, exist_ok=True)
        _log.info('Directory %s created', fig_dir)
        for file in os.listdir(fig_dir):
            full_path = os.path.join(fig_dir, file)
            try:
                os.remove(full_path)
            except OSError:
                _log.info('Failed to remove %s', full_path)
    return fig_dir