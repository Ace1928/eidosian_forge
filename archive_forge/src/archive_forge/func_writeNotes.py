from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import vex_util
def writeNotes(self, notes, project, uri):
    notes_to_create = []
    notes_to_update = []
    notes_to_retain = notes
    for note in notes:
        get_request = self.ca_messages.ContaineranalysisProjectsNotesGetRequest(name='projects/{}/notes/{}'.format(project, note.key))
        try:
            self.ca_client.projects_notes.Get(get_request)
            note_exists = True
        except apitools_exceptions.HttpNotFoundError:
            note_exists = False
        if note_exists:
            notes_to_update.append(note)
        else:
            notes_to_create.append(note)
    self.batchWriteNotes(notes_to_create, project)
    self.updateNotes(notes_to_update, project)
    self.deleteNotes(notes_to_retain, project, uri)