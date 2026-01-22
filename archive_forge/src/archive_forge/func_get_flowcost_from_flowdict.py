import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def get_flowcost_from_flowdict(G, flowDict):
    """Returns flow cost calculated from flow dictionary"""
    flowCost = 0
    for u in flowDict:
        for v in flowDict[u]:
            flowCost += flowDict[u][v] * G[u][v]['weight']
    return flowCost