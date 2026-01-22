def combined_commuting_positive_definite_hint(operator_a, operator_b):
    """Get combined PD hint for compositions."""
    if operator_a.is_positive_definite is True and operator_a.is_self_adjoint is True and (operator_b.is_positive_definite is True) and (operator_b.is_self_adjoint is True):
        return True
    return None