import uuid
from keystoneclient.tests.unit.v3 import utils
Test project-endpoint associations (a.k.a. EndpointFilter Extension).

    Endpoint filter provides associations between service endpoints and
    projects. These assciations are then used to create ad-hoc catalogs for
    each project-scoped token request.

    