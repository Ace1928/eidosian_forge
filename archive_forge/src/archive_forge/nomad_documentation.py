import os
import os.path as op
import subprocess
Upload files to NOMAD.

    Upload all data within specified folders to the Nomad repository
    using authentication token given by the --token option or,
    if no token is given, the token stored in ~/.ase/nomad-token.

    To get an authentication token, you create a Nomad repository account
    and use the 'Uploads' button on that page while logged in:

      https://repository.nomad-coe.eu/
    