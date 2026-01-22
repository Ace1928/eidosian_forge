from boto.cloudsearch2.optionstatus import IndexFieldStatus
from boto.cloudsearch2.optionstatus import ServicePoliciesStatus
from boto.cloudsearch2.optionstatus import ExpressionStatus
from boto.cloudsearch2.optionstatus import AvailabilityOptionsStatus
from boto.cloudsearch2.optionstatus import ScalingParametersStatus
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.search import SearchConnection
def create_expression(self, name, value):
    """
        Create a new expression.

        :type name: string
        :param name: The name of an expression for processing
            during a search request.

        :type value: string
        :param value: The expression to evaluate for ranking
            or thresholding while processing a search request. The
            Expression syntax is based on JavaScript expressions
            and supports:

            * Single value, sort enabled numeric fields (int, double, date)
            * Other expressions
            * The _score variable, which references a document's relevance
              score
            * The _time variable, which references the current epoch time
            * Integer, floating point, hex, and octal literals
            * Arithmetic operators: + - * / %
            * Bitwise operators: | & ^ ~ << >> >>>
            * Boolean operators (including the ternary operator): && || ! ?:
            * Comparison operators: < <= == >= >
            * Mathematical functions: abs ceil exp floor ln log2 log10 logn
             max min pow sqrt pow
            * Trigonometric functions: acos acosh asin asinh atan atan2 atanh
             cos cosh sin sinh tanh tan
            * The haversin distance function

            Expressions always return an integer value from 0 to the maximum
            64-bit signed integer value (2^63 - 1). Intermediate results are
            calculated as double-precision floating point values and the return
            value is rounded to the nearest integer. If the expression is
            invalid or evaluates to a negative value, it returns 0. If the
            expression evaluates to a value greater than the maximum, it
            returns the maximum value.

            The source data for an Expression can be the name of an
            IndexField of type int or double, another Expression or the
            reserved name _score. The _score source is
            defined to return as a double from 0 to 10.0 (inclusive) to
            indicate how relevant a document is to the search request,
            taking into account repetition of search terms in the
            document and proximity of search terms to each other in
            each matching IndexField in the document.

            For more information about using rank expressions to
            customize ranking, see the Amazon CloudSearch Developer
            Guide.

        :return: ExpressionStatus object
        :rtype: :class:`boto.cloudsearch2.option.ExpressionStatus` object

        :raises: BaseException, InternalException, LimitExceededException,
            InvalidTypeException, ResourceNotFoundException
        """
    data = self.layer1.define_expression(self.name, name, value)
    data = data['DefineExpressionResponse']['DefineExpressionResult']['Expression']
    return ExpressionStatus(self, data, self.layer1.describe_expressions)