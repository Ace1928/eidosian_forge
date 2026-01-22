import logging
def existential_quantifier(predicate, domain):
    return any((predicate(x) for x in domain))