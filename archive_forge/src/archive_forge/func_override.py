from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def override(cls):
    """Decorator for documenting method overrides.

    Args:
        cls: The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.

    .. testcode::
        :skipif: True

        from ray.rllib.policy import Policy
        class TorchPolicy(Policy):
            ...
            # Indicates that `TorchPolicy.loss()` overrides the parent
            # Policy class' own `loss method. Leads to an error if Policy
            # does not have a `loss` method.

            @override(Policy)
            def loss(self, model, action_dist, train_batch):
                ...

    """

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError('{} does not override any method of {}'.format(method, cls))
        return method
    return check_override