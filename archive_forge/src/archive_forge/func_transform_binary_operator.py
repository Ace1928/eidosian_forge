from sympy.external import import_module
import os
def transform_binary_operator(self, node):
    """Transformation function for handling binary operators

            Returns
            =======

            binary_expression: Codegen AST node
                    simplified binary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If a bitwise operator or
                unary operator(which is a child of any binary
                operator in Clang AST) is encountered

            """
    tokens = list(node.get_tokens())
    operators_list = ['+', '-', '*', '/', '%', '=', '>', '>=', '<', '<=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=']
    combined_variables_stack = []
    operators_stack = []
    for token in tokens:
        if token.kind == cin.TokenKind.PUNCTUATION:
            if token.spelling == '(':
                operators_stack.append('(')
            elif token.spelling == ')':
                while operators_stack and operators_stack[-1] != '(':
                    if len(combined_variables_stack) < 2:
                        raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
                    rhs = combined_variables_stack.pop()
                    lhs = combined_variables_stack.pop()
                    operator = operators_stack.pop()
                    combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
                operators_stack.pop()
            elif token.spelling in operators_list:
                while operators_stack and self.priority_of(token.spelling) <= self.priority_of(operators_stack[-1]):
                    if len(combined_variables_stack) < 2:
                        raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
                    rhs = combined_variables_stack.pop()
                    lhs = combined_variables_stack.pop()
                    operator = operators_stack.pop()
                    combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
                operators_stack.append(token.spelling)
            elif token.spelling in ['&', '|', '^', '<<', '>>']:
                raise NotImplementedError('Bitwise operator has not been implemented yet!')
            elif token.spelling in ['&=', '|=', '^=', '<<=', '>>=']:
                raise NotImplementedError('Shorthand bitwise operator has not been implemented yet!')
            else:
                raise NotImplementedError('Given token {} is not implemented yet!'.format(token.spelling))
        elif token.kind == cin.TokenKind.IDENTIFIER:
            combined_variables_stack.append([token.spelling, 'identifier'])
        elif token.kind == cin.TokenKind.LITERAL:
            combined_variables_stack.append([token.spelling, 'literal'])
        elif token.kind == cin.TokenKind.KEYWORD and token.spelling in ['true', 'false']:
            combined_variables_stack.append([token.spelling, 'boolean'])
        else:
            raise NotImplementedError('Given token {} is not implemented yet!'.format(token.spelling))
    while operators_stack:
        if len(combined_variables_stack) < 2:
            raise NotImplementedError('Unary operators as a part of binary operators is not supported yet!')
        rhs = combined_variables_stack.pop()
        lhs = combined_variables_stack.pop()
        operator = operators_stack.pop()
        combined_variables_stack.append(self.perform_operation(lhs, rhs, operator))
    return combined_variables_stack[-1][0]