from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import collections.abc
import os
def is_explicit_method(self):
    """Determines if a C++ constructor or conversion function is
        explicit, returning 1 if such is the case and 0 otherwise.

        Constructors or conversion functions are declared explicit through
        the use of the explicit specifier.

        For example, the following constructor and conversion function are
        not explicit as they lack the explicit specifier:

            class Foo {
                Foo();
                operator int();
            };

        While the following constructor and conversion function are
        explicit as they are declared with the explicit specifier.

            class Foo {
                explicit Foo();
                explicit operator int();
            };

        This method will return 0 when given a cursor pointing to one of
        the former declarations and it will return 1 for a cursor pointing
        to the latter declarations.

        The explicit specifier allows the user to specify a
        conditional compile-time expression whose value decides
        whether the marked element is explicit or not.

        For example:

            constexpr bool foo(int i) { return i % 2 == 0; }

            class Foo {
                 explicit(foo(1)) Foo();
                 explicit(foo(2)) operator int();
            }

        This method will return 0 for the constructor and 1 for
        the conversion function.
        """
    return conf.lib.clang_CXXMethod_isExplicit(self)