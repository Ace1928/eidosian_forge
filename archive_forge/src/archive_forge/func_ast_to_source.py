import re
import copy
import inspect
import ast
import textwrap
def ast_to_source(ast):
    """Convert AST to source code string using the astor package"""
    import astor
    return astor.to_source(ast)