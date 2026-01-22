import sys
import os.path as op
import pyxnat
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_002_delete_experiment():
    print('DELETING')
    e = 'BBRCDEV_E03099'
    cols = ['subject_label', 'label']
    e0 = central.array.experiments(experiment_id=e, columns=cols).data[0]
    subject_label = e0['subject_label']
    experiment_label = e0['label']
    e1 = central.array.experiments(project_id=dest_project, subject_label=subject_label, experiment_label=experiment_label, columns=['subject_id']).data[0]
    p = central.select.project(dest_project)
    e2 = p.subject(e1['subject_ID']).experiment(e1['ID'])
    assert e2.exists()
    e2.delete()
    assert not e2.exists()