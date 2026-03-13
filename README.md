# Ferrite Fraction Prediction

Predict ferrite fraction from HEXRD 2-theta profiles using a pre-trained model.

## Quick start

1) Install dependencies:
```sh
pip install -r requirements.txt
```

2) Get the model file:

Option A: Git LFS (recommended)
```sh
git lfs install
git lfs pull
```

Option B: Direct download (no Git LFS)
```
https://github.com/benrabah2/ferrite_fraction_predictor/blob/main/model_v1.0.0.h5
```
Place the file in the repo folder.

3) Set your model and defaults in `config.py`:
- Update `model_name` to point to your `.h5` file.
- Optional defaults you can change: `Convert_to_q`, `shape_correction`, `Normalization`, `nb_of_points_per_profile`, `Q_MIN`, `Q_MAX`, `batch_size`, `plot_data`.

4) Run:
```sh
python predict.py /path/to/your/data_folder -w WAVELENGTH -s START -e END
```
If you omit `-s/--start` and `-e/--end`, the script will use the min and max index found in the folder:
```sh
python predict.py /path/to/your/data_folder -w WAVELENGTH
```
The script writes results to `data_folder/results/predicted_ferrite_fraction_<DATASET_NAME>.txt`.
If `-n/--name` is not provided, it uses the folder name as `<DATASET_NAME>` for the output file.

Optional overrides:
- `--q-min`, `--q-max`
- `--no-convert-to-q`, `--no-shape-correction`, `--no-normalization`
- `--nb-points`, `--batch-size`
- `--plot`
- `-n/--name` (to filter by dataset prefix)
- `-s/--start` and `-e/--end` are optional; if omitted, the script uses the min and max index found in the folder.
- `--out-dir`, `--out-name`, `--out-path`

Recommendation:
- Prefer using `-n`, `-s`, and `-e` to avoid mixing datasets and to control the range explicitly.

## Data format

Each data file must be a text file with at least two columns:
- column 1: 2-theta
- column 2: intensity

File naming (two modes):
- If you provide `-n/--name`, filenames must start with that name.
- After the name, any non-digit separator(s) are allowed (`_`, `-`, `.`, etc.).
- Filenames must end with a numeric index and `.dat`.
- If you do **not** provide `-n/--name`, all `.dat` files in the folder that end with a numeric index are used.

Examples accepted with `-n SAMPLE`:
- `SAMPLE_00001.dat`
- `SAMPLE-00001.dat`
- `SAMPLE__00001.dat`
- `SAMPLE.anything.00001.dat`

Example not accepted (digits in separator):
- `SAMPLE_v2_00001.dat`

## Output

- Default output folder: `<data_folder>/results`
- Default output file name: `predicted_ferrite_fraction_<DATASET_NAME>.txt`
- Use `--out-dir` to change the output folder.
- Use `--out-name` to change the output file name.
- Use `--out-path` to specify a full output path (highest priority).

## Model file

This repository uses Git LFS for the model file.
If you clone without Git LFS, you will only get a small pointer file, not the real `.h5`.
To ensure the model downloads:
```sh
git lfs install
git lfs pull
```

If you do not want to use Git LFS, download the model directly and place it in the repo folder:
```
https://github.com/benrabah2/ferrite_fraction_predictor/blob/main/model_v1.0.0.h5
```
Then make sure `config.py` points to `model_v1.0.0.h5`.

Model versioning:
- Prefer semantic versioning for model files, e.g., `model_v1.0.0.h5`.

## What the script does

- Loads each profile in the configured index range.
- Optionally converts 2-theta to q, applies shape correction, and normalizes (single normalization step inside model prediction).
- Runs the model to predict ferrite fraction per file.
- Optionally plots the results.

## Citation

If you use this code, please cite:

Benrabah, Imed‐Eddine, Guillaume Geandier, Olha Nakonechna, Benoît Denand, Hugo Van Landeghem, Alexis Deschamps, and Sébastien YP Allain. "Deep Learning for Real‐Time Phase Quantification from X‐ray Diffraction: Toward High‐Throughput Steel Microstructure Mapping." *Advanced Engineering Materials* (2026): e202503172. https://doi.org/10.1002/adem.202503172

Please also reference the model GitHub repository: https://github.com/benrabah2/ferrite_fraction_predictor

## Project files

- `predict.py`: command-line entry point.
- `data_utils.py`: preprocessing functions.
- `model_utils.py`: model loading and prediction.
- `config.py`: model path and preprocessing defaults.
