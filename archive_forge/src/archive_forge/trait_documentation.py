from osc_lib.command import command
from osc_placement import version
Dissociate all the traits from the resource provider.

    Note that this command is not atomic if multiple processes are managing
    traits for the same provider.

    This command requires at least ``--os-placement-api-version 1.6``.
    