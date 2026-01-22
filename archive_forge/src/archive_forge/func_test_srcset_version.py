import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def test_srcset_version(tmp_path):
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path, dirs_exist_ok=True)
    html_dir = tmp_path / '_build' / 'html'
    img_dir = html_dir / '_images'
    doctree_dir = tmp_path / 'doctrees'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=['-D', 'plot_srcset=2x'])

    def plot_file(num, suff=''):
        return img_dir / f'some_plots-{num}{suff}.png'
    for ind in [1, 2, 3, 5, 7, 11, 13, 15, 17]:
        assert plot_file(ind).exists()
        assert plot_file(ind, suff='.2x').exists()
    assert (img_dir / 'nestedpage-index-1.png').exists()
    assert (img_dir / 'nestedpage-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage-index-2.png').exists()
    assert (img_dir / 'nestedpage-index-2.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-1.png').exists()
    assert (img_dir / 'nestedpage2-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-2.png').exists()
    assert (img_dir / 'nestedpage2-index-2.2x.png').exists()
    assert 'srcset="_images/some_plots-1.png, _images/some_plots-1.2x.png 2.00x"' in (html_dir / 'some_plots.html').read_text(encoding='utf-8')
    st = 'srcset="../_images/nestedpage-index-1.png, ../_images/nestedpage-index-1.2x.png 2.00x"'
    assert st in (html_dir / 'nestedpage/index.html').read_text(encoding='utf-8')
    st = 'srcset="../_images/nestedpage2-index-2.png, ../_images/nestedpage2-index-2.2x.png 2.00x"'
    assert st in (html_dir / 'nestedpage2/index.html').read_text(encoding='utf-8')