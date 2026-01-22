import hvplot
import pandas as pd
import pytest
def test_fugure_ipython_line(table, capsys):
    """hvplot works with Fugue"""
    fa.fugue_sql('\n        OUTPUT table USING hvplot:line(\n            x="x",\n            y="y",\n            by="g",\n            size=100,\n            opts={"width": 500, "height": 500}\n        )\n        ')
    output = capsys.readouterr().out
    assert output == 'Column\n    [0] HoloViews(NdOverlay)\n'