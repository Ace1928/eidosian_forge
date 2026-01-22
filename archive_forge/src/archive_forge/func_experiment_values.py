from . import schema
from .jsonutil import get_column
from .search import Search
def experiment_values(self, datatype, project=None):
    """ Look for the values a the experiment level for a given datatype
            in the database.

            .. note::
                The  datatype should be one of Inspector.experiment_types()

            Parameters
            ----------
            datatype: string
                An experiment type. eg: xnat:mrsessiondata
            project: string
                Optional. Restrict operation to a project.
        """
    self._intf._get_entry_point()
    uri = '%s/experiments?columns=ID' % self._intf._entry
    if datatype is not None:
        uri += '&xsiType=%s' % datatype
    if project is not None:
        uri += '&project=%s' % project
    return get_column(self._get_json(uri), 'ID')