import pytest
from shapely import Point, Polygon
from shapely.geos import geos_version
def test_format_point():
    xy1 = (0.12345678901234566, 12345678901.234568)
    xy2 = (-169.910918, -18.997564)
    xyz3 = (630084, 4833438, 76)
    test_list = [('.0f', xy1, 'POINT (0 12345678901)', True), ('.1f', xy1, 'POINT (0.1 12345678901.2)', True), ('0.2f', xy2, 'POINT (-169.91 -19.00)', True), ('.3F', (float('inf'), -float('inf')), 'POINT (INF -INF)', True)]
    if geos_version < (3, 10, 0):
        test_list += [('.1g', xy1, 'POINT (0.1 1e+10)', True), ('.6G', xy1, 'POINT (0.123457 1.23457E+10)', True), ('0.12g', xy1, 'POINT (0.123456789012 12345678901.2)', True), ('0.4g', xy2, 'POINT (-169.9 -19)', True)]
    else:
        test_list += [('.1g', xy1, 'POINT (0.1 12345678901.2)', False), ('.6G', xy1, 'POINT (0.123457 12345678901.234568)', False), ('0.12g', xy1, 'POINT (0.123456789012 12345678901.234568)', False), ('g', xy2, 'POINT (-169.910918 -18.997564)', False), ('0.2g', xy2, 'POINT (-169.91 -19)', False)]
    test_list += [('f', (1, 2), f'POINT ({1:.16f} {2:.16f})', False), ('F', xyz3, 'POINT Z ({:.16f} {:.16f} {:.16f})'.format(*xyz3), False), ('g', xyz3, 'POINT Z (630084 4833438 76)', False)]
    for format_spec, coords, expt_wkt, same_python_float in test_list:
        pt = Point(*coords)
        assert f'{pt}' == pt.wkt
        assert format(pt, '') == pt.wkt
        assert format(pt, 'x') == pt.wkb_hex.lower()
        assert format(pt, 'X') == pt.wkb_hex
        assert format(pt, format_spec) == expt_wkt, format_spec
        text_coords = expt_wkt[expt_wkt.index('(') + 1:expt_wkt.index(')')]
        is_same = []
        for coord, expt_coord in zip(coords, text_coords.split()):
            py_fmt_float = format(float(coord), format_spec)
            if same_python_float:
                assert py_fmt_float == expt_coord, format_spec
            else:
                is_same.append(py_fmt_float == expt_coord)
        if not same_python_float:
            assert not all(is_same), f'{format_spec!r} with {expt_wkt}'