from autograd.extend import primitive, defvjp, vspace
from autograd.builtins import tuple
from autograd import make_vjp
def rev_iter(params):
    a, x_star, x_star_bar = params
    vjp_x, _ = make_vjp(f(a))(x_star)
    vs = vspace(x_star)
    return lambda g: vs.add(vjp_x(g), x_star_bar)