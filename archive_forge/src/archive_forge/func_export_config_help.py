import argparse
from textwrap import dedent
import pandas
import modin.config as cfg
def export_config_help(filename: str) -> None:
    """
    Export all configs help messages to the CSV file.

    Parameters
    ----------
    filename : str
        Name of the file to export configs data.
    """
    configs_data = []
    default_values = dict(RayRedisPassword='random string', CpuCount='multiprocessing.cpu_count()', NPartitions='equals to MODIN_CPUS env')
    for objname in sorted(cfg.__all__):
        obj = getattr(cfg, objname)
        if isinstance(obj, type) and issubclass(obj, cfg.Parameter) and (not obj.is_abstract):
            data = {'Config Name': obj.__name__, 'Env. Variable Name': getattr(obj, 'varname', 'not backed by environment'), 'Default Value': default_values.get(obj.__name__, obj._get_default()), 'Description': dedent(obj.__doc__ or '').replace('Notes\n-----', 'Notes:\n'), 'Options': obj.choices}
            configs_data.append(data)
    pandas.DataFrame(configs_data, columns=['Config Name', 'Env. Variable Name', 'Default Value', 'Description', 'Options']).to_csv(filename, index=False)