# ND2-Processor

Batch-processes Nikon `.nd2` microscopy files into labeled, color-rendered
JPEGs per channel, plus DAPI-based cell measurements (area, length, width,
aspect ratio, circularity) exported to CSV, with an annotated overlay image.

## What it does

For each `.nd2` file, the script (`nd2_processor_script.py`):

1. Reads all timepoints/positions and channels via `ND2Reader`.
2. Classifies each channel by name (DAPI/405, FITC/488, TRITC/561,
   brightfield, or generic) and renders it:
   - **DAPI** — percentile-normalized, gamma-corrected, false-colored red.
   - **FITC** — percentile-normalized, gamma-corrected, false-colored green.
   - **TRITC** — percentile-normalized, gamma-corrected, false-colored magenta/red.
   - **Brightfield** — min/max normalized + CLAHE contrast enhancement.
   - **Other/unknown** — simple percentile-stretched grayscale.
3. Draws a scale bar, filename/channel labels, and a processing timestamp
   onto each rendered image, then saves it as a `.jpg`.
4. If a DAPI channel is present, segments nuclei/cells by thresholding,
   removes small objects, labels connected regions, and measures each
   region (area in µm², length/width in µm, aspect ratio, circularity).
   Measurements are written to a `_cells.csv` file, and an annotated JPEG
   with per-cell outlines and stat labels is saved.

Output for each input file `sample.nd2` is written to `Output/sample/`,
containing per-channel, per-timepoint JPEGs, a `_cells.csv`, and an
`_annotated.jpg` per timepoint (when DAPI is present).

## Requirements

- Python 3.9+
- Packages: `nd2reader`, `opencv-python`, `numpy`, `scikit-image`, `pandas`

Install with:

```bash
pip install nd2reader opencv-python numpy scikit-image pandas
```

## Usage

Process every `.nd2` file in the default input folder:

```bash
python nd2_processor_script.py
```

Process every `.nd2` file in a custom input/output folder:

```bash
python nd2_processor_script.py --input "C:\path\to\input" --output "C:\path\to\output"
```

Process a single file:

```bash
python nd2_processor_script.py --file "C:\path\to\file.nd2"
```

### Arguments

| Argument   | Default                              | Description                                  |
|------------|---------------------------------------|-----------------------------------------------|
| `--input`  | `C:\ICARE_APPs\ND2-Script\Input`      | Folder containing `.nd2` files to process     |
| `--output` | `C:\ICARE_APPs\ND2-Script\Output`     | Folder where results are written              |
| `--file`   | *(none)*                              | Process a single `.nd2` file instead of a folder |

## Notes

- Channel type is inferred from the channel name (case-insensitive
  substring match on `dapi`/`405`, `fitc`/`488`, `tritc`/`561`,
  `bf`/`bright`); unmatched channels are rendered as grayscale.
- Cell segmentation uses a simple 99th-percentile threshold on the DAPI
  channel and drops objects smaller than 50 px — tune `measure_cells()` if
  your data needs different sensitivity.
- Pixel size (µm/px) is read from the ND2 metadata (`pixel_microns`) and
  falls back to `1.0` if unavailable, which affects scale bars and
  physical-unit measurements.
