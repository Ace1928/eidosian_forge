import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
@pytest.mark.parametrize('font_str', ["'DejaVu Sans', 'WenQuanYi Zen Hei', 'Arial', sans-serif", "'DejaVu Serif', 'WenQuanYi Zen Hei', 'Times New Roman', serif", "'Arial', 'WenQuanYi Zen Hei', cursive", "'Impact', 'WenQuanYi Zen Hei', fantasy", "'DejaVu Sans Mono', 'WenQuanYi Zen Hei', 'Courier New', monospace"])
@pytest.mark.parametrize('include_generic', [True, False])
def test_svg_font_string(font_str, include_generic):
    fp = fm.FontProperties(family=['WenQuanYi Zen Hei'])
    if Path(fm.findfont(fp)).name != 'wqy-zenhei.ttc':
        pytest.skip('Font may be missing')
    explicit, *rest, generic = map(lambda x: x.strip("'"), font_str.split(', '))
    size = len(generic)
    if include_generic:
        rest = rest + [generic]
    plt.rcParams[f'font.{generic}'] = rest
    plt.rcParams['font.size'] = size
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots()
    if generic == 'sans-serif':
        generic_options = ['sans', 'sans-serif', 'sans serif']
    else:
        generic_options = [generic]
    for generic_name in generic_options:
        ax.text(0.5, 0.5, 'There are 几个汉字 in between!', family=[explicit, generic_name], ha='center')
        ax.text(0.5, 0.1, 'There are 几个汉字 in between!', family=[explicit, *rest, generic_name], ha='center')
    ax.axis('off')
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue()
    tree = xml.etree.ElementTree.fromstring(buf)
    ns = 'http://www.w3.org/2000/svg'
    text_count = 0
    for text_element in tree.findall(f'.//{{{ns}}}text'):
        text_count += 1
        font_info = dict((map(lambda x: x.strip(), _.strip().split(':')) for _ in dict(text_element.items())['style'].split(';')))['font']
        assert font_info == f'{size}px {font_str}'
    assert text_count == len(ax.texts)