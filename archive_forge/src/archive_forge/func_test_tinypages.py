import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def test_tinypages(tmp_path):
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path, dirs_exist_ok=True)
    html_dir = tmp_path / '_build' / 'html'
    img_dir = html_dir / '_images'
    doctree_dir = tmp_path / 'doctrees'
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html', '-d', str(doctree_dir), str(Path(__file__).parent / 'tinypages'), str(html_dir)]
    proc = subprocess_run_for_testing(cmd, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': '', 'GCOV_ERROR_FILE': os.devnull})
    out = proc.stdout
    err = proc.stderr
    build_sphinx_html(tmp_path, doctree_dir, html_dir)

    def plot_file(num):
        return img_dir / f'some_plots-{num}.png'

    def plot_directive_file(num):
        return doctree_dir.parent / 'plot_directive' / f'some_plots-{num}.png'
    range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]
    assert filecmp.cmp(range_6, plot_file(5))
    assert filecmp.cmp(range_4, plot_file(7))
    assert filecmp.cmp(range_10, plot_file(11))
    assert filecmp.cmp(range_10, plot_file('12_00'))
    assert filecmp.cmp(range_6, plot_file('12_01'))
    assert filecmp.cmp(range_4, plot_file(13))
    html_contents = (html_dir / 'some_plots.html').read_bytes()
    assert b'# Only a comment' in html_contents
    assert filecmp.cmp(range_4, img_dir / 'range4.png')
    assert filecmp.cmp(range_6, img_dir / 'range6_range6.png')
    assert b'This is the caption for plot 15.' in html_contents
    assert b'Plot 17 uses the caption option.' in html_contents
    assert b'This is the caption for plot 18.' in html_contents
    assert b'plot-directive my-class my-other-class' in html_contents
    assert html_contents.count(b'This caption applies to both plots.') == 2
    assert filecmp.cmp(range_6, plot_file(17))
    assert filecmp.cmp(range_10, img_dir / 'range6_range10.png')
    contents = (tmp_path / 'included_plot_21.rst').read_bytes()
    contents = contents.replace(b'plt.plot(range(6))', b'plt.plot(range(4))')
    (tmp_path / 'included_plot_21.rst').write_bytes(contents)
    modification_times = [plot_directive_file(i).stat().st_mtime for i in (1, 2, 3, 5)]
    build_sphinx_html(tmp_path, doctree_dir, html_dir)
    assert filecmp.cmp(range_4, plot_file(17))
    assert plot_directive_file(1).stat().st_mtime == modification_times[0]
    assert plot_directive_file(2).stat().st_mtime == modification_times[1]
    assert plot_directive_file(3).stat().st_mtime == modification_times[2]
    assert filecmp.cmp(range_10, plot_file(1))
    assert filecmp.cmp(range_6, plot_file(2))
    assert filecmp.cmp(range_4, plot_file(3))
    assert plot_directive_file(5).stat().st_mtime > modification_times[3]
    assert filecmp.cmp(range_6, plot_file(5))