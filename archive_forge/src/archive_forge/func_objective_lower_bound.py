from sys import version_info as _swig_python_version_info
import weakref
def objective_lower_bound(self):
    """
        Returns the current lower bound found by internal solvers during the
        search.
        """
    return _pywrapcp.RoutingModel_objective_lower_bound(self)