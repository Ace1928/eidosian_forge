def missing_codec_lib(codec, *libraries):

    def missing(fo):
        raise ValueError(f'{codec} codec is supported but you need to install one of the ' + f'following libraries: {libraries}')
    return missing