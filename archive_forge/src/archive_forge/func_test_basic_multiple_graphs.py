import pytest
import networkx as nx
def test_basic_multiple_graphs():
    H1 = nx.path_graph(4)
    H2 = nx.complete_graph(4)
    H3 = nx.path_graph(8)
    H4 = nx.complete_graph(8)
    captions = ['Path on 4 nodes', 'Complete graph on 4 nodes', 'Path on 8 nodes', 'Complete graph on 8 nodes']
    labels = ['fig2a', 'fig2b', 'fig2c', 'fig2d']
    latex_code = nx.to_latex([H1, H2, H3, H4], n_rows=2, sub_captions=captions, sub_labels=labels)
    assert 'begin{document}' in latex_code
    assert 'begin{figure}' in latex_code
    assert latex_code.count('begin{subfigure}') == 4
    assert latex_code.count('tikzpicture') == 8
    assert latex_code.count('[-]') == 4