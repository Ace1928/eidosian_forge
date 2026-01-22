from nltk.sem.logic import *
def compute_substitution_semantics(function, argument):
    assert isinstance(function, LambdaExpression) and isinstance(function.term, LambdaExpression), '`' + str(function) + '` must be a lambda expression with 2 arguments'
    assert isinstance(argument, LambdaExpression), '`' + str(argument) + '` must be a lambda expression'
    new_argument = ApplicationExpression(argument, VariableExpression(function.variable)).simplify()
    new_term = ApplicationExpression(function.term, new_argument).simplify()
    return LambdaExpression(function.variable, new_term)