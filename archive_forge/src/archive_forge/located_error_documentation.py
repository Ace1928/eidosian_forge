from typing import TYPE_CHECKING, Collection, Optional, Union
from ..pyutils import inspect
from .graphql_error import GraphQLError
Located GraphQL Error

    Given an arbitrary Exception, presumably thrown while attempting to execute a
    GraphQL operation, produce a new GraphQLError aware of the location in the document
    responsible for the original Exception.
    