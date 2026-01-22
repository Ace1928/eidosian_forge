def listify(fun):
    """ Decorator to make a function apply
    to each item in a sequence, and return a list. """

    def f(seq):
        res = [fun(x) for x in seq]
        return res
    return f