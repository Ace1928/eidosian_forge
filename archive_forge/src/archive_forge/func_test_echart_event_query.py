from panel.layout import Row
from panel.pane import ECharts, Markdown
def test_echart_event_query(document, comm):
    echart = ECharts(ECHART, width=500, height=500)
    echart.on_event('click', print, 'series.line')
    model = echart.get_root(document, comm)
    assert model.data == ECHART
    assert model.event_config == {'click': ['series.line']}