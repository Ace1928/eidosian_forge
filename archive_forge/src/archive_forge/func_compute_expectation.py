from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution
def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
    """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
    return self.process.expectation(expr, condition, evaluate, **kwargs)