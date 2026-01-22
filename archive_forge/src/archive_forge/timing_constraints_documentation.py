from qiskit.transpiler.exceptions import TranspilerError
Initialize a TimingConstraints object

        Args:
            granularity: An integer value representing minimum pulse gate
                resolution in units of ``dt``. A user-defined pulse gate should have
                duration of a multiple of this granularity value.
            min_length: An integer value representing minimum pulse gate
                length in units of ``dt``. A user-defined pulse gate should be longer
                than this length.
            pulse_alignment: An integer value representing a time resolution of gate
                instruction starting time. Gate instruction should start at time which
                is a multiple of the alignment value.
            acquire_alignment: An integer value representing a time resolution of measure
                instruction starting time. Measure instruction should start at time which
                is a multiple of the alignment value.

        Notes:
            This information will be provided by the backend configuration.

        Raises:
            TranspilerError: When any invalid constraint value is passed.
        