Calculate the dwell time

    Parameters
    ----------
    water_fat_shift : float
        The water fat shift of the recording, in pixels.
    echo_train_length : int
        The echo train length of the imaging sequence.
    field_strength : float
        Strength of the magnet in Tesla, e.g. 3.0 for a 3T magnet recording.

    Returns
    -------
    dwell_time : float
        The dwell time in seconds.

    Raises
    ------
    MRIError
        if values are out of range
    