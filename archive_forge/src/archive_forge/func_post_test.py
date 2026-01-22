import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
def post_test(test, parsed):
    parsed_stack = exprStack[:]
    exprStack.clear()
    name, testcase, expected = next((tc for tc in testcases if tc[1] == test))
    this_test_passed = False
    try:
        try:
            result = _evaluateStack(parsed_stack)
        except TypeError:
            print("Unsupported operation on right side of '%s'.\nCheck for missing or incorrect tags on non-scalar operands." % input_string, file=sys.stderr)
            raise
        except UnaryUnsupportedError:
            print("Unary negation is not supported for vectors and matrices: '%s'" % input_string, file=sys.stderr)
            raise
        if debug_flag:
            print('var=', targetvar)
        if targetvar != None:
            try:
                result = _assignfunc(targetvar, result)
            except TypeError:
                print("Left side tag does not match right side of '%s'" % input_string, file=sys.stderr)
                raise
            except UnaryUnsupportedError:
                print("Unary negation is not supported for vectors and matrices: '%s'" % input_string, file=sys.stderr)
                raise
        else:
            print("Empty left side in '%s'" % input_string, file=sys.stderr)
            raise TypeError
        parsed['result'] = result
        parsed['passed'] = this_test_passed = result == expected
    finally:
        all_passed[0] = all_passed[0] and this_test_passed
        print('\n' + name)