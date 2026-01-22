from ..objecttype import ObjectType, Field
from ..scalars import Scalar, Int, BigInt, Float, String, Boolean
from ..schema import Schema
from graphql import Undefined
from graphql.language.ast import IntValueNode
def return_input(_parent, _info, input):
    return input