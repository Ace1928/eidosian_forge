def build_anomalies_description(self):
    """builds descriptions for the anomalies"""
    d = self.events_definition
    anomalies_description = {evname: d[evname]['desc'] for evname in self.bads}
    anomalies_description['scheduling_in_past'] = 'Scheduling in the past'
    return anomalies_description