from __future__ import print_function
import argparse
import sys
import graphviz
from ._discover import findMachines
def makeDigraph(automaton, inputAsString=repr, outputAsString=repr, stateAsString=repr):
    """
    Produce a L{graphviz.Digraph} object from an automaton.
    """
    digraph = graphviz.Digraph(graph_attr={'pack': 'true', 'dpi': '100'}, node_attr={'fontname': 'Menlo'}, edge_attr={'fontname': 'Menlo'})
    for state in automaton.states():
        if state is automaton.initialState:
            stateShape = 'bold'
            fontName = 'Menlo-Bold'
        else:
            stateShape = ''
            fontName = 'Menlo'
        digraph.node(stateAsString(state), fontame=fontName, shape='ellipse', style=stateShape, color='blue')
    for n, eachTransition in enumerate(automaton.allTransitions()):
        inState, inputSymbol, outState, outputSymbols = eachTransition
        thisTransition = 't{}'.format(n)
        inputLabel = inputAsString(inputSymbol)
        port = 'tableport'
        table = tableMaker(inputLabel, [outputAsString(outputSymbol) for outputSymbol in outputSymbols], port=port)
        digraph.node(thisTransition, label=_gvhtml(table), margin='0.2', shape='none')
        digraph.edge(stateAsString(inState), '{}:{}:w'.format(thisTransition, port), arrowhead='none')
        digraph.edge('{}:{}:e'.format(thisTransition, port), stateAsString(outState))
    return digraph