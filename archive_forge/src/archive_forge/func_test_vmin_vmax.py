import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_vmin_vmax(self):
    df = self.world.copy()
    df['range'] = range(len(df))
    m = df.explore('range', vmin=-100, vmax=1000)
    out_str = self._fetch_map_string(m)
    assert 'case"176":return{"color":"#3b528b","fillColor":"#3b528b"' in out_str
    assert 'case"119":return{"color":"#414287","fillColor":"#414287"' in out_str
    assert 'case"3":return{"color":"#482173","fillColor":"#482173"' in out_str
    df2 = self.nybb.copy()
    df2['values'] = df2['BoroCode'] * 10.0
    m = df2[df2['values'] >= 30].explore('values', vmin=0)
    out_str = self._fetch_map_string(m)
    if FOLIUM_G_014:
        assert 'case"0":return{"color":"#fde725","fillColor":"#fde725"' in out_str
        assert 'case"1":return{"color":"#7ad151","fillColor":"#7ad151"' in out_str
        assert 'default:return{"color":"#22a884","fillColor":"#22a884"' in out_str
    else:
        assert 'case"1":return{"color":"#7ad151","fillColor":"#7ad151"' in out_str
        assert 'case"2":return{"color":"#22a884","fillColor":"#22a884"' in out_str
        assert 'default:return{"color":"#fde725","fillColor":"#fde725"' in out_str
    df2['values_negative'] = df2['BoroCode'] * -10.0
    m = df2[df2['values_negative'] <= 30].explore('values_negative', vmax=0)
    out_str = self._fetch_map_string(m)
    assert 'case"1":return{"color":"#414487","fillColor":"#414487"' in out_str
    assert 'case"2":return{"color":"#2a788e","fillColor":"#2a788e"' in out_str