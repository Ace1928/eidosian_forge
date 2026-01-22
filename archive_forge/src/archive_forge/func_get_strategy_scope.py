import json
import os
import tensorflow.compat.v2 as tf
def get_strategy_scope(strategy):
    if strategy:
        strategy_scope = strategy.scope()
    else:
        strategy_scope = DummyContextManager()
    return strategy_scope