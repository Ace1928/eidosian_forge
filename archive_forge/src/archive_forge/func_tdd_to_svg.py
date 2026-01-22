from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def tdd_to_svg(tdd: 'cirq.TextDiagramDrawer', ref_boxwidth: float=40, ref_boxheight: float=40, col_padding: float=20, row_padding: float=10) -> str:
    row_starts, row_heights, yi_map = _fit_vertical(tdd=tdd, ref_boxheight=ref_boxheight, row_padding=row_padding)
    col_starts, col_widths = _fit_horizontal(tdd=tdd, ref_boxwidth=ref_boxwidth, col_padding=col_padding)
    t = f'<svg xmlns="http://www.w3.org/2000/svg" width="{col_starts[-1]}" height="{row_starts[-1]}">'
    for yi, xi1, xi2, _, _ in tdd.horizontal_lines:
        xi1 = cast(int, xi1)
        xi2 = cast(int, xi2)
        x1 = col_starts[xi1] + col_widths[xi1] / 2
        x2 = col_starts[xi2] + col_widths[xi2] / 2
        yi = yi_map[yi]
        y = row_starts[yi] + row_heights[yi] / 2
        if xi1 == 0:
            stroke = QBLUE
        else:
            stroke = 'black'
        t += f'<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="{stroke}" stroke-width="1" />'
    for xi, yi1, yi2, _, _ in tdd.vertical_lines:
        yi1 = yi_map[yi1]
        yi2 = yi_map[yi2]
        y1 = row_starts[yi1] + row_heights[yi1] / 2
        y2 = row_starts[yi2] + row_heights[yi2] / 2
        xi = cast(int, xi)
        x = col_starts[xi] + col_widths[xi] / 2
        t += f'<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" stroke="black" stroke-width="3" />'
    for (xi, yi), v in tdd.entries.items():
        yi = yi_map[yi]
        x = col_starts[xi] + col_widths[xi] / 2
        y = row_starts[yi] + row_heights[yi] / 2
        boxheight = ref_boxheight
        boxwidth = col_widths[xi] - tdd.horizontal_padding.get(xi, col_padding)
        boxx = x - boxwidth / 2
        boxy = y - boxheight / 2
        if xi == 0:
            t += _rect(boxx, boxy, boxwidth, boxheight, strokewidth=0)
            t += _text(x, y, v.text)
            continue
        if v.text == '@':
            t += f'<circle cx="{x}" cy="{y}" r="{ref_boxheight / 4}" />'
            continue
        if v.text == '×':
            t += _text(x, y + 3, '×', fontsize=40)
            continue
        if v.text == '':
            continue
        v_text = fixup_text(v.text)
        t += _rect(boxx, boxy, boxwidth, boxheight)
        t += _text(x, y, v_text, fontsize=14 if len(v_text) > 1 else 18)
    t += '</svg>'
    return t