from .jsonutil import JsonTable
def mrscans(self, project_id=None, subject_id=None, subject_label=None, experiment_id=None, experiment_label=None, columns=None, constraints=None):
    """ Returns a list of all MR scans, filtered by optional constraints.

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
            columns: list
                Values to return.
            constraints: dict
                Dictionary of xsi_type (key--) and parameter (--value)
                pairs by which to filter.
            """
    return self.scans(project_id, subject_id, subject_label, experiment_id, experiment_label, 'xnat:mrSessionData', 'xnat:mrScanData', columns, constraints)