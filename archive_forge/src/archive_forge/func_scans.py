from .jsonutil import JsonTable
def scans(self, project_id=None, subject_id=None, subject_label=None, experiment_id=None, experiment_label=None, experiment_type='xnat:imageSessionData', scan_type='xnat:imageScanData', columns=None, constraints=None):
    """ Returns a list of all visible scan IDs of the specified type,
            filtered by optional constraints.

            Parameters
            ----------
            project_id: string
                Name pattern to filter by project ID.
            subject_id: string
                Name pattern to filter by subject ID.
            subject_label: string
                Name pattern to filter by subject ID.
            experiment_id: string
                Name pattern to filter by experiment ID.
            experiment_label: string
                Name pattern to filter by experiment ID.
            experiment_type: string
                xsi path type; e.g. 'xnat:mrSessionData'
            scan_type: string
                xsi path type; e.g. 'xnat:mrScanData', etc.
            columns: list
                Values to return.
            constraints: dict
                Dictionary of xsi_type (key--) and parameter (--value)
                pairs by which to filter.
            """
    query_string = '&columns=ID,project,%s/subject_id,%s/ID' % (experiment_type, scan_type)
    array = self._get_array(query_string, project_id, subject_id, subject_label, experiment_id, experiment_label, experiment_type, columns, constraints)
    id_key = ('%s/ID' % scan_type).lower()
    return JsonTable([i for i in array if i[id_key]])