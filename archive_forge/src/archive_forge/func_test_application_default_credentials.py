import os
import google.auth
def test_application_default_credentials(verify_refresh):
    credentials, project_id = google.auth.default()
    if EXPECT_PROJECT_ID is not None:
        assert project_id is not None
    verify_refresh(credentials)