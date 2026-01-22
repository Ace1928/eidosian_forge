from __future__ import absolute_import, division, print_function
from ray import Language
from ray._raylet import CppFunctionDescriptor, JavaFunctionDescriptor
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def cpp_actor_class(create_function_name: str, class_name: str):
    """Define a Cpp actor class.

    Args:
        create_function_name: Create cpp class function name.
        class_name: Cpp class name.
    """
    from ray.actor import ActorClass
    print('create func=', create_function_name, 'class_name=', class_name)
    return ActorClass._ray_from_function_descriptor(Language.CPP, CppFunctionDescriptor(create_function_name, 'PYTHON', class_name), {})