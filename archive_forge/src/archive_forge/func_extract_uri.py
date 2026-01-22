from .schema import rest_translation
def extract_uri(uri):
    """
    Destructure the given REST uri into project,subject and experiment.

    Returns None if any one of project,subject or experiment is unspecified in
    the URI and a (project,subject,experiment) triple otherwise.
    """
    split = uri.split('/')
    if len(split) != 9:
        return None
    project = split[3]
    subject = split[5]
    experiment = split[7]
    return (project, subject, experiment)