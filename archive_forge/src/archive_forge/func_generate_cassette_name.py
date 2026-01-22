import json
import os
import betamax.serializers.base
import yaml
@staticmethod
def generate_cassette_name(cassette_library_dir, cassette_name):
    return os.path.join(cassette_library_dir, '{name}.yaml'.format(name=cassette_name))