from pathlib import Path
import io
import pytest
from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
@pytest.mark.parametrize('family_name, file_name', [('WenQuanYi Zen Hei', 'wqy-zenhei'), ('Noto Sans CJK JP', 'NotoSansCJK')])
@check_figures_equal(extensions=['png', 'pdf', 'eps', 'svg'])
def test_font_fallback_chinese(fig_test, fig_ref, family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    if file_name not in Path(fm.findfont(fp)).name:
        pytest.skip(f'Font {family_name} ({file_name}) is missing')
    text = ['There are', '几个汉字', 'in between!']
    plt.rcParams['font.size'] = 20
    test_fonts = [['DejaVu Sans', family_name]] * 3
    ref_fonts = [['DejaVu Sans'], [family_name], ['DejaVu Sans']]
    for j, (txt, test_font, ref_font) in enumerate(zip(text, test_fonts, ref_fonts)):
        fig_ref.text(0.05, 0.85 - 0.15 * j, txt, family=ref_font)
        fig_test.text(0.05, 0.85 - 0.15 * j, txt, family=test_font)