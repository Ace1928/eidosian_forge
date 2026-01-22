from panel.layout import Row
from panel.pane import ECharts, Markdown
def test_echart(document, comm):
    echart = ECharts(ECHART, width=500, height=500)
    model = echart.get_root(document, comm)
    assert model.data == ECHART