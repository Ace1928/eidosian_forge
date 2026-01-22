import warnings
import entrypoints
def register_entrypoints(self):
    for entrypoint in entrypoints.get_group_all(REQUEST_AUTH_PROVIDER_ENTRYPOINT):
        try:
            self.register(entrypoint.load())
        except (AttributeError, ImportError) as exc:
            warnings.warn('Failure attempting to register request auth provider "{}": {}'.format(entrypoint.name, str(exc)), stacklevel=2)