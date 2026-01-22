from sympy.external import import_module
def visit_Variable(self, node):
    """Visitor Function for Variable Declaration

            Visits each variable declaration present in the ASR and creates a
            Symbol declaration for each variable

            Notes
            =====

            The functions currently only support declaration of integer and
            real variables. Other data types are still under development.

            Raises
            ======

            NotImplementedError() when called for unsupported data types

            """
    if isinstance(node.type, asr.Integer):
        var_type = IntBaseType(String('integer'))
        value = Integer(0)
    elif isinstance(node.type, asr.Real):
        var_type = FloatBaseType(String('real'))
        value = Float(0.0)
    else:
        raise NotImplementedError('Data type not supported')
    if not node.intent == 'in':
        new_node = Variable(node.name).as_Declaration(type=var_type, value=value)
        self._py_ast.append(new_node)