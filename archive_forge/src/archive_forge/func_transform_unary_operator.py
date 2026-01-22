from sympy.external import import_module
import os
def transform_unary_operator(self, node):
    """Transformation function for handling unary operators

            Returns
            =======

            unary_expression: Codegen AST node
                    simplified unary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If dereferencing operator(*), address operator(&) or
                bitwise NOT operator(~) is encountered

            """
    operators_list = ['+', '-', '++', '--', '!']
    tokens = list(node.get_tokens())
    if tokens[0].spelling in operators_list:
        child = self.transform(next(node.get_children()))
        if isinstance(child, str):
            if tokens[0].spelling == '+':
                return Symbol(child)
            if tokens[0].spelling == '-':
                return Mul(Symbol(child), -1)
            if tokens[0].spelling == '++':
                return PreIncrement(Symbol(child))
            if tokens[0].spelling == '--':
                return PreDecrement(Symbol(child))
            if tokens[0].spelling == '!':
                return Not(Symbol(child))
        else:
            if tokens[0].spelling == '+':
                return child
            if tokens[0].spelling == '-':
                return Mul(child, -1)
            if tokens[0].spelling == '!':
                return Not(sympify(bool(child)))
    elif tokens[1].spelling in ['++', '--']:
        child = self.transform(next(node.get_children()))
        if tokens[1].spelling == '++':
            return PostIncrement(Symbol(child))
        if tokens[1].spelling == '--':
            return PostDecrement(Symbol(child))
    else:
        raise NotImplementedError('Dereferencing operator, Address operator and bitwise NOT operator have not been implemented yet!')