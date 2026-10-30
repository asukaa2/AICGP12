def run_mdx(model_params, output_dir, model_name, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process the audio
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    roformer_output_format = 'wav'
    roformer_overlap = 4
    roformer_segment_size = 256
    print(f"output_dir: {output_dir}")
    prompt = f'audio-separator "{filename}" --model_filename {model_name} --output_dir="{output_dir}" --output_format={roformer_output_format} --normalization=0.9 --mdxc_overlap={roformer_overlap} --mdxc_segment_size={roformer_segment_size}'
    os.system(prompt)

    vocals_file = f"{base_name}_Vocals.wav"
    instrumental_file = f"{base_name}_Instrumental.wav"

    main_filepath = None
    invert_filepath = None

    if not exclude_main:
        main_filepath = os.path.join(output_dir, vocals_file)
        if os.path.exists(os.path.join(output_dir, f"{base_name}_(Vocals)_{model_name.replace('.onnx', '')}.wav")):
            os.rename(os.path.join(output_dir, f"{base_name}_(Vocals)_{model_name.replace('.onnx', '')}.wav"), main_filepath)

    if not exclude_inversion:
        invert_filepath = os.path.join(output_dir, instrumental_file)
        if os.path.exists(os.path.join(output_dir, f"{base_name}_(Instrumental)_{model_name.replace('.pth', '')}.wav")):
            os.rename(os.path.join(output_dir, f"{base_name}_(Instrumental)_{model_name.replace('.pth', '')}.wav"), invert_filepath)

    if not keep_orig:
        os.remove(filename)

    return main_filepath, invert_filepath

def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    device_properties = torch.cuda.get_device_properties(device)
    vram_gb = device_properties.total_memory / 1024**3
    m_threads = 1 if vram_gb < 8 else 2

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )

    mdx_sess = MDX(model_path, model)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return main_filepath, invert_filepath



