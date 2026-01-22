import logging
import xml.etree.ElementTree as ET
from fiona.env import require_gdal_version
from fiona.ogrext import _get_metadata_item
@require_gdal_version('2.0')
def print_driver_options(driver):
    """ Print driver options for dataset open, dataset creation, and layer creation.

    Parameters
    ----------
    driver : str

    """
    for option_type, options in [('Dataset Open Options', dataset_open_options(driver)), ('Dataset Creation Options', dataset_creation_options(driver)), ('Layer Creation Options', layer_creation_options(driver))]:
        print(f'{option_type}:')
        if len(options) == 0:
            print('\tNo options available.')
        else:
            for option_name in options:
                print(f'\t{option_name}:')
                if 'description' in options[option_name]:
                    print('\t\tDescription: {description}'.format(description=options[option_name]['description']))
                if 'type' in options[option_name]:
                    print('\t\tType: {type}'.format(type=options[option_name]['type']))
                if 'values' in options[option_name] and len(options[option_name]['values']) > 0:
                    print('\t\tAccepted values: {values}'.format(values=','.join(options[option_name]['values'])))
                for attr_text, attribute in [('Default value', 'default'), ('Required', 'required'), ('Alias', 'aliasOf'), ('Min', 'min'), ('Max', 'max'), ('Max size', 'maxsize'), ('Scope', 'scope'), ('Alternative configuration option', 'alt_config_option')]:
                    if attribute in options[option_name]:
                        print('\t\t{attr_text}: {attribute}'.format(attr_text=attr_text, attribute=options[option_name][attribute]))
        print('')