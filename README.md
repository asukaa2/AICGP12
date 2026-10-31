<div align="center">

# AICoverGen (Modded)

An autonomous pipeline to create AI song covers with any RVC v2 trained voice —
from a YouTube link or a local audio file.

Built for developers who want to add singing to their AI assistant, chatbot, or
vtuber, and for anyone who wants to hear their favourite characters sing their
favourite song.

Modded & maintained by **[AsukaDev](https://github.com/asukaa2)** ·
Forked from [SociallyIneptWeeb/AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asukaa2/AICGP12/blob/main/AICoverGen_colab.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Showcase](https://www.youtube.com/watch?v=2qZuE4WM7CM)** ·
**[Setup Guide](https://www.youtube.com/watch?v=pdlhk4vVHQk)**

![](images/webui_generate.png?raw=true)

</div>

---

The WebUI is under active development, but you can try it out right now on both
local and Colab. The pipeline runs end-to-end with no manual intervention: pull
audio → split vocals/instrumentals → split main/backup vocals → de-reverb →
pitch-shift → RVC voice conversion → reverb/mixing → final cover.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [WebUI](#usage-with-webui)
  - [CLI](#usage-with-cli)
  - [Scripted (`no_ui.py`)](#usage-with-the-scripted-runner-no_uipy)
- [Configuration Reference](#configuration-reference)
- [Running locally on Python 3.8 (no Colab)](#running-locally-on-python-38-no-colab)
- [Repository Layout](#repository-layout)
- [Terms of Use](#terms-of-use)
- [Disclaimer](#disclaimer)

## Features

**Pipeline**

- One-shot cover generation from a YouTube URL **or** a local audio file
- Automatic vocal/instrumental separation (roformer when `audio-separator` is
  installed, falls back to the bundled MDX-Net `UVR-MDX-NET-Voc_FT.onnx`)
- Automatic main/backup vocal split (`UVR_MDXNET_KARA_2.onnx`)
- De-reverb pass on main vocals (`Reverb_HQ_By_FoxJoy.onnx`)
- RVC v2 voice conversion with optional `.index` retrieval
- Two pitch-extraction backends: `rmvpe` (default, clearer) and `mangio-crepe`
  (smoother), plus experimental `hybrid` mode
- Optional overall key change (vocals + instrumentals together, in semitones)
- Reverb control on AI vocals (room size, wetness, dryness, damping)
- Independent volume control for main vocals, backup vocals, and instrumentals
- Output as `wav` (best quality) or `mp3` (small size)
- Option to keep intermediate files (isolated vocals/instrumentals/etc.)

**WebUI**

- Generate tab with all main and advanced options surfaced in collapsible
  accordions
- Download model tab — paste a HuggingFace / Pixeldrain / Google Drive link
  to a `.zip` and the model is extracted into `rvc_models/`
- Public model index — a searchable, tag-filterable table of pre-trained voice
  models; click a row to auto-fill the download link
- Upload model tab — drop a locally trained RVC v2 `.zip` to register it
- Refresh button to pick up models added manually to `rvc_models/`
- `--share` for a public gradio URL (handy on Colab)
- `--listen` to expose the UI on your local network

**RVC conversion options**

- `index_rate` — how much of the AI voice's accent to keep
- `filter_radius` — median filtering on harvested pitch (0–7)
- `rms_mix_rate` — original loudness vs fixed loudness (0–1)
- `protect` — preserve breath/voiceless consonants (0–0.5)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/asukaa2/AICGP12
cd AICGP12

# 2. Install deps (Python 3.9+)
pip install -r requirements.txt

# 3. Download the HuBERT base model + rmvpe model
python src/download_models.py

# 4. Run the WebUI
python src/webui.py
```

Open `http://127.0.0.1:7860` in your browser. If you have no local NVIDIA GPU,
use the [Colab notebook](https://colab.research.google.com/github/asukaa2/AICGP12/blob/main/AICoverGen_colab.ipynb)
instead.

> 💡 On Python 3.8, `pip install -r requirements.txt` will fail due to a
> dependency clash between `fairseq` and `audio-separator`. Use the
> [Python 3.8 setup](#running-locally-on-python-38-no-colab) below instead.

## Installation

> **Note:** If `pip install -r requirements.txt` fails, it is not your machine.
> The default requirements can't all be installed on a single Python version
> — see [Running locally on Python 3.8 (no Colab)](#running-locally-on-python-38-no-colab).

### 1. Install Git and Python

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git),
then install **Python 3.9** following
[this guide](https://realpython.com/installing-python/). Other Python versions
may cause dependency conflicts.

### 2. Install ffmpeg

Follow [this guide](https://www.hostinger.com/tutorials/how-to-install-ffmpeg)
to install ffmpeg.

### 3. Install sox

Follow
[this guide](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/)
to install sox and add it to your `PATH`.

### 4. Clone & install

```bash
git clone https://github.com/asukaa2/AICGP12
cd AICGP12
pip install -r requirements.txt
```

### 5. Download required models

```bash
python src/download_models.py
```

This pulls the HuBERT base model (`rvc_models/hubert_base.pt`) and the rmvpe
pitch model (`rvc_models/rmvpe.pt`) from the official RVC HuggingFace repo.
MDX-Net separation models are looked up by hash in
`mdxnet_models/model_data.json` and loaded on demand.

## Usage

### Usage with WebUI

```bash
python src/webui.py
```

| Flag                              | Description                                                            |
|----------------------------------|------------------------------------------------------------------------|
| `-h`, `--help`                    | Show the help message and exit.                                        |
| `--share`                         | Create a public URL. Useful for running the WebUI on Google Colab.     |
| `--listen`                        | Make the WebUI reachable from your local network.                      |
| `--listen-host LISTEN_HOST`       | The hostname the server will use.                                      |
| `--listen-port LISTEN_PORT`       | The listening port the server will use.                                |

When you see `Running on local URL: http://127.0.0.1:7860`, click the link to
open the WebUI.

#### Download RVC models via WebUI

![](images/webui_dl_model.png?raw=true)

Navigate to the **Download model** tab, paste the download link to an RVC
model `.zip`, and give it a unique name. You can search the
[AI Hub Discord](https://discord.gg/aihub) for already trained voice models —
refer to the examples in the tab for what the link should look like. The
downloaded `.zip` should contain a `.pth` model file and an optional `.index`
file.

When the output message says `[NAME] Model successfully downloaded!`, click
the refresh button and the model will appear in the **Generate** tab.

#### Upload RVC models via WebUI

![](images/webui_upload_model.png?raw=true)

If you have trained RVC v2 models locally, navigate to the **Upload model**
tab and follow the instructions. When the output message says
`[NAME] Model successfully uploaded!`, click refresh and use it in the
**Generate** tab.

#### Running the pipeline via WebUI

![](images/webui_generate.png?raw=true)

- From the **Voice Models** dropdown, select the voice model to use. Click
  `Refresh Models 🔁` if you added files manually to [rvc_models](rvc_models).
- In the **Song input** field, paste a YouTube link or the full path to a
  local audio file. Use the `Upload file instead` button to upload a file
  through the browser.
- Set **Pitch Change** to `-12`, `0`, or `12` depending on the original vocals
  and the RVC AI model. This keeps the voice from being *out of tune*.
- Advanced options for voice conversion and audio mixing are in collapsible
  accordions.

Click **Generate**. The AI cover appears in a few minutes, depending on your
GPU. Two outputs are produced:

- **AI Cover (Vocal Only Inference)** — main AI vocals mixed with the original
  instrumentals and original backup vocals
- **AI Cover (Vocal Backing Inference)** — main AI vocals mixed with AI-converted
  backup vocals and the original instrumentals

### Usage with CLI

#### Manual download of RVC models

Unzip (if needed) and place the `.pth` and `.index` files into a new folder
under [rvc_models](rvc_models). Each folder should contain exactly one `.pth`
and at most one `.index` file.

```
AICGP12/
├── rvc_models/
│   ├── John/
│   │   ├── JohnV2.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── May/
│   │   ├── May.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── MODELS.txt
│   └── hubert_base.pt
├── mdxnet_models/
├── song_output/
└── src/
```

#### Running the pipeline

```bash
python src/main.py \
  -i SONG_INPUT \
  -dir RVC_DIRNAME \
  -p PITCH_CHANGE \
  [-k | --keep-files | --no-keep-files] \
  [-ir INDEX_RATE] [-fr FILTER_RADIUS] [-rms RMS_MIX_RATE] \
  [-palgo PITCH_DETECTION_ALGO] [-hop CREPE_HOP_LENGTH] [-pro PROTECT] \
  [-mv MAIN_VOL] [-bv BACKUP_VOL] [-iv INST_VOL] \
  [-pall PITCH_CHANGE_ALL] \
  [-rsize REVERB_SIZE] [-rwet REVERB_WETNESS] [-rdry REVERB_DRYNESS] [-rdamp REVERB_DAMPING] \
  [-oformat OUTPUT_FORMAT]
```

| Flag                            | Default | Description |
|---------------------------------|---------|-------------|
| `-h`, `--help`                  | —       | Show the help message and exit. |
| `-i SONG_INPUT`                 | —       | Link to a song on YouTube, or the filepath to a local `mp3`/`wav`. Use double quotes on Windows, single quotes on Unix. |
| `-dir MODEL_DIR_NAME`           | —       | Name of the folder in [rvc_models](rvc_models) containing your `.pth` and optional `.index` files. |
| `-p PITCH_CHANGE`               | —       | Change pitch of AI vocals in **octaves**. `0` = no change; `1` for male→female, `-1` for female→male. |
| `-k` / `--keep-files`           | off     | Keep all intermediate audio files in `song_output/<id>/` (isolated vocals/instrumentals, etc.). |
| `-ir INDEX_RATE`                | 0.5     | `0–1`. How much of the AI voice's accent to leave in. `1` biases more towards the training dataset timbre. |
| `-fr FILTER_RADIUS`             | 3       | `0–7`. If `≥3`, apply median filtering to the harvested pitch results; reduces breathiness. |
| `-rms RMS_MIX_RATE`             | 0.25    | `0–1`. `0` mimics the original vocal's loudness, `1` uses a fixed loudness. |
| `-palgo PITCH_DETECTION_ALGO`   | `rmvpe` | `rmvpe` (clarity) or `mangio-crepe` (smoother). Also supports `hybrid` mode in code. |
| `-hop CREPE_HOP_LENGTH`         | 128     | Only used with `mangio-crepe`. Lower = better pitch accuracy but slower and higher risk of voice cracks. |
| `-pro PROTECT`                  | 0.33    | `0–0.5`. Preserves breath/voiceless consonants. `0.5` disables protection. |
| `-mv MAIN_VOCALS_VOLUME_CHANGE` | 0       | Decibels. `-3` reduces main AI vocals by 3 dB, `3` boosts by 3 dB. |
| `-bv BACKUP_VOCALS_VOLUME_CHANGE` | 0     | Decibels. Same semantics as `-mv` for backup vocals. |
| `-iv INSTRUMENTAL_VOLUME_CHANGE`| 0       | Decibels. Same semantics for instrumentals. |
| `-pall PITCH_CHANGE_ALL`        | 0       | **Semitones.** Shifts the key of vocals + instrumentals together. Slightly reduces sound quality. |
| `-rsize REVERB_SIZE`            | 0.15    | `0–1`. Larger room = longer reverb tail. |
| `-rwet REVERB_WETNESS`          | 0.2     | `0–1`. Level of AI vocals with reverb. |
| `-rdry REVERB_DRYNESS`          | 0.8     | `0–1`. Level of AI vocals without reverb. |
| `-rdamp REVERB_DAMPING`         | 0.7     | `0–1`. Absorption of high frequencies in the reverb. |
| `-oformat OUTPUT_FORMAT`        | `mp3`   | `mp3` (small, decent quality) or `wav` (large, best quality). |

**Examples**

```bash
# YouTube link, default everything, mp3 out
python src/main.py -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -dir John -p 0

# Local file, keep intermediates, shift key up 2 semitones, wav out
python src/main.py -i ./song.wav -dir John -p 0 -k -pall 2 -oformat wav

# Tighten accent with index_rate=0.8, smoother vocals with mangio-crepe
python src/main.py -i ./song.wav -dir John -p -1 -ir 0.8 -palgo mangio-crepe -hop 64
```

### Usage with the scripted runner (`no_ui.py`)

`no_ui.py` is a thin wrapper that prompts you for a YouTube URL and forwards
everything else to `src/main.py`. Edit the defaults at the top of the file,
then:

```bash
python no_ui.py
# Enter Youtube URL: <paste>
```

It streams `main.py`'s stdout live to your terminal.

## Configuration Reference

### Directory layout

| Path                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `rvc_models/`         | One folder per voice. Each folder holds a `.pth` + optional `.index`. Also holds `hubert_base.pt` and `rmvpe.pt`. |
| `mdxnet_models/`      | MDX-Net separation models (`.onnx`) + `model_data.json` (hash → params). |
| `song_output/`       | Generated covers and intermediate audio, one subfolder per `song_id`.  |
| `src/configs/`        | RVC model configs per sample rate (`32k`/`40k`/`48k`, plus `_v2`).      |
| `src/infer_pack/`     | RVC synthesizer models, attention, commons, modules.                    |
| `images/`             | WebUI screenshots used in this README.                                  |

### Separation backend selection

The first split (vocals vs. instrumentals) auto-selects its backend:

- If `audio-separator` is installed (Python 3.9+) → uses the higher-quality
  roformer model `model_bs_roformer_ep_317_sdr_12.9755.ckpt` via the
  `audio-separator` CLI (`run_roformer`).
- Otherwise → falls back to the bundled MDX-Net `UVR-MDX-NET-Voc_FT.onnx`
  loaded in-process with ONNX Runtime (`run_mdx`).

The backup/main split (`UVR_MDXNET_KARA_2.onnx`) and the de-reverb pass
(`Reverb_HQ_By_FoxJoy.onnx`) always use the in-process MDX path.

### Pitch detection

- `rmvpe` (default) — fast, clear vocals. Requires `rvc_models/rmvpe.pt`.
- `mangio-crepe` — smoother vocals, slower. Use with `-hop` to control
  accuracy vs. speed.
- `hybrid` — combine multiple methods by median (advanced; set as a
  string in code).

## Running locally on Python 3.8 (no Colab)

If `pip install -r requirements.txt` fails, it is not your machine — the
default requirements can't all be installed on a single Python version:

- `fairseq==0.12.2` only has a prebuilt wheel for Python 3.8 and older. On
  3.11/3.12 pip tries to build it from source and fails on newer
  CPython/Cython — that is the `egg_info` build error most people hit.
- `scipy>=1.13`, `audio-separator`, and `argparse.BooleanOptionalAction`
  need Python 3.9 or newer.

So you can't have both. The setup below pins everything to Python 3.8 and
runs the whole pipeline locally. Tested end to end on an RTX 2060
(6 GB, CUDA 11.8): it generates a cover from a local audio file with the
GPU doing the work (~5 GB VRAM at peak).

```bash
# 1. Python 3.8 venv
python3.8 -m venv venv && source venv/bin/activate
pip install pip==23.3.1 wheel setuptools

# 2. Torch with CUDA 11.8
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Everything else (fairseq's cp38 wheel installs cleanly, no source build)
pip install -r requirements_py38.txt

# 4. CUDA 11.8 runtime that onnxruntime-gpu needs for MDX-Net separation
pip install nvidia-cudnn-cu11==8.7.0.84

# 5. Download models, then run on a local file
python src/download_models.py
python src/main.py -i path/to/song.wav -dir YourModelFolder -p 0 -k -oformat mp3
```

One thing to know: `audio-separator` doesn't run on Python 3.8, so the first
vocal/instrumental split uses the bundled MDX-Net model
(`UVR-MDX-NET-Voc_FT.onnx`) instead. The pipeline picks this automatically.
On Python 3.9+ with `audio-separator` installed you get the higher-quality
roformer split — nothing changes for that setup.

## Repository Layout

```
AICGP12/
├── src/
│   ├── main.py                      # CLI entrypoint + song_cover_pipeline
│   ├── webui.py                     # Gradio WebUI (Generate / Download / Upload tabs)
│   ├── rvc.py                       # Config, HuBERT loader, RVC model loader, infer
│   ├── pipeline.py                  # VC.pipeline — feature extraction + voice conversion
│   ├── mdx.py                       # MDX-Net ONNX runner + audio-separator CLI runner
│   ├── rmvpe.py                     # RMVPE pitch model
│   ├── fairseq.py                   # Bundled fairseq shim (load_model) for HuBERT
│   ├── my_utils.py                  # ffmpeg-based audio loader
│   ├── download_models.py           # Pulls hubert_base.pt + rmvpe.pt
│   ├── trainset_preprocess_pipeline_print.py
│   ├── configs/                     # RVC model configs per sample rate
│   └── infer_pack/                  # Synthesizer models, attention, modules, commons
├── rvc_models/                      # Voice model folders + hubert_base.pt + rmvpe.pt
├── mdxnet_models/                   # MDX-Net .onnx models + model_data.json
├── song_output/                     # Generated covers (created on first run)
├── AICoverGen_colab.ipynb           # One-click Colab notebook
├── no_ui.py                         # Interactive prompt wrapper around main.py
├── requirements.txt                 # Python 3.9+ deps
├── requirements_py38.txt            # Python 3.8 deps (no audio-separator)
├── images/                          # WebUI screenshots
├── LICENSE
└── README.md
```

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

- Criticizing or attacking individuals.
- Advocating for or opposing specific political positions, religions, or
  ideologies.
- Publicly displaying strongly stimulating expressions without proper zoning.
- Selling of voice models and generated voice clips.
- Impersonation of the original owner of the voice with malicious intentions
  to harm/hurt others.
- Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or
special damages arising out of or in any way connected with the use/misuse or
inability to use this software.
