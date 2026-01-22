from breezy import errors, tests, workingtree
Ensure that BadSubsumeSource is raised.

        SubsumeTargetNeedsUpgrade must not be raised, because upgrading the
        target won't help.
        