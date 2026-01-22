from pathlib import Path
import io
import pytest
from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
def test_fallback_smoke():
    fp = fm.FontProperties(family=['WenQuanYi Zen Hei'])
    if Path(fm.findfont(fp)).name != 'wqy-zenhei.ttc':
        pytest.skip('Font wqy-zenhei.ttc may be missing')
    fp = fm.FontProperties(family=['Noto Sans CJK JP'])
    if Path(fm.findfont(fp)).name != 'NotoSansCJK-Regular.ttc':
        pytest.skip('Noto Sans CJK JP font may be missing.')
    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(4.75, 1.85))
    fig.text(0.05, 0.45, 'There are 几个汉字 in between!', family=['DejaVu Sans', 'Noto Sans CJK JP'])
    fig.text(0.05, 0.25, 'There are 几个汉字 in between!', family=['DejaVu Sans', 'WenQuanYi Zen Hei'])
    fig.text(0.05, 0.65, 'There are 几个汉字 in between!', family=['Noto Sans CJK JP'])
    fig.text(0.05, 0.85, 'There are 几个汉字 in between!', family=['WenQuanYi Zen Hei'])
    for fmt in ['png', 'raw']:
        fig.savefig(io.BytesIO(), format=fmt)