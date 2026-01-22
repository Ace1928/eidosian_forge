from pyomo.repn.standard_repn import (

    This plugin computes the standard representation for all objectives
    and constraints. All results are stored in a ComponentMap named
    "_repn" at the block level.

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    