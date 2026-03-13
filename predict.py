import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from model_utils import load_model, predict_ferrite_fraction, predict_ferrite_fraction_batched
from data_utils import Convert_2theta_to_q, Shape_Correction_Function
from config import (
    nb_of_points_per_profile,
    Normalization,
    Convert_to_q,
    shape_correction,
    model_name,
    Q_MIN,
    Q_MAX,
    plot_data,
    batch_size,
    output_dir_name,
    output_name_template,
)


def _get_dataset_config(name, start, end, wave_length):
    if wave_length is None:
        return name, start, end, wave_length
    return name, start, end, wave_length


def _load_profile(file_name, wave_length):
    data = np.loadtxt(file_name)[:, :2]
    if Convert_to_q:
        if wave_length is None:
            raise ValueError("Wavelength is required when converting 2-theta to q. Use -w/--wavelength or pass --no-convert-to-q.")
        data = Convert_2theta_to_q(data, wave_length, Q_MIN, Q_MAX)
    if shape_correction:
        data = Shape_Correction_Function(data, nb_of_points_per_profile)
    return data


def _extract_index_from_filename(fname, dataset_name=None):
    if not fname.lower().endswith(".dat"):
        return None
    if dataset_name:
        if not fname.startswith(dataset_name):
            return None
        remainder = fname[len(dataset_name):]
        match = re.match(r"\D+(\d+)\.dat$", remainder)
    else:
        match = re.search(r"(\d+)\.dat$", fname)
    if not match:
        return None
    return int(match.group(1))


def _list_candidate_files(folder_name, dataset_name):
    candidates = []
    for fname in os.listdir(folder_name):
        idx = _extract_index_from_filename(fname, dataset_name=dataset_name)
        if idx is None:
            continue
        candidates.append((idx, fname))
    return candidates


def _collect_profiles(folder_name, dataset_name, start, end, wave_length):
    profiles = []
    file_numbers = []

    candidates = _list_candidate_files(folder_name, dataset_name)
    if not candidates:
        return profiles, file_numbers

    all_indices = [idx for idx, _ in candidates]
    min_idx, max_idx = min(all_indices), max(all_indices)
    if start is None:
        start = min_idx
    if end is None:
        end = max_idx

    for idx, fname in sorted(candidates, key=lambda x: x[0]):
        if idx < start or idx > end:
            continue
        file_name = os.path.join(folder_name, fname)
        print(file_name)
        if not os.path.isfile(file_name):
            continue
        profiles.append(_load_profile(file_name, wave_length))
        file_numbers.append(float(idx))

    return profiles, file_numbers


def _resolve_output_path(data_folder, dataset_name, out_dir=None, out_name=None, out_path=None):
    if out_path:
        return out_path
    if out_dir:
        base_dir = out_dir if os.path.isabs(out_dir) else os.path.join(data_folder, out_dir)
    else:
        base_dir = os.path.join(data_folder, output_dir_name)
    os.makedirs(base_dir, exist_ok=True)
    filename = out_name if out_name else output_name_template.format(dataset=dataset_name)
    return os.path.join(base_dir, filename)


def _save_predictions(
    data_folder,
    dataset_name,
    file_numbers,
    ferrite_fraction,
    wave_length,
    out_dir=None,
    out_name=None,
    out_path=None,
):
    output_file = _resolve_output_path(
        data_folder,
        dataset_name,
        out_dir=out_dir,
        out_name=out_name,
        out_path=out_path,
    )
    with open(output_file, 'w') as f:
        f.write("# Ferrite fraction prediction\n")
        f.write(f"# model_name: {model_name}\n")
        f.write(f"# convert_to_q: {Convert_to_q}\n")
        f.write(f"# q_range: {Q_MIN} to {Q_MAX}\n")
        f.write(f"# wavelength: {wave_length}\n")
        f.write(f"# timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("File Number\tPredicted Ferrite Fraction (%)\n")
        for file_number, ferrite_frac in zip(file_numbers, ferrite_fraction):
            f.write(f"{file_number}\t{ferrite_frac:.4f}\n")
    print(f"Predicted ferrite fraction data saved to {output_file}")


def main(data_folder, name, start, end, wave_length, out_dir=None, out_name=None, out_path=None):
    name, start, end, wave_length = _get_dataset_config(name, start, end, wave_length)
    if name:
        print(name)

    profiles, file_numbers = _collect_profiles(data_folder, name, start, end, wave_length)
    if not profiles:
        dataset_label = name if name else "any dataset"
        raise ValueError(f"No data files found in {data_folder} for {dataset_label}.")

    X = np.stack(profiles)
    X_2d = X.reshape(X.shape[0], -1)
    if file_numbers:
        print(f"Loaded {len(file_numbers)} profiles (index {int(file_numbers[0])} to {int(file_numbers[-1])}).")

    ferrite_fraction = predict_ferrite_fraction_batched(
        X_2d,
        model_name,
        batch_size=batch_size,
        apply_normalization=Normalization,
    )

    if plot_data:
        plt.plot(file_numbers, ferrite_fraction, '-.', markersize=8, color='green')
        plt.ylabel('Predicted Ferrite fraction (%)', fontsize=12)
        plt.xlabel('File number', fontsize=12)
        plt.show()

    dataset_label = name if name else os.path.basename(os.path.abspath(data_folder))
    _save_predictions(
        data_folder,
        dataset_label,
        file_numbers,
        ferrite_fraction,
        wave_length,
        out_dir=out_dir,
        out_name=out_name,
        out_path=out_path,
    )
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for ferrite fraction prediction.')
    parser.add_argument('data_folder', type=str, help='Directory containing the data files')
    parser.add_argument('-n', '--name', type=str, required=False, help='Dataset name used for file prefix (optional)')
    parser.add_argument('-w', '--wavelength', type=float, required=False, help='Wavelength for the dataset (required if converting 2-theta to q)')
    parser.add_argument('-s', '--start', type=int, required=False, help='Start index for the dataset (optional)')
    parser.add_argument('-e', '--end', type=int, required=False, help='End index for the dataset (optional)')
    parser.add_argument('--q-min', type=float, default=Q_MIN, help='Minimum q value (default from config)')
    parser.add_argument('--q-max', type=float, default=Q_MAX, help='Maximum q value (default from config)')
    parser.add_argument('--no-convert-to-q', action='store_true', help='Disable 2-theta to q conversion')
    parser.add_argument('--no-shape-correction', action='store_true', help='Disable shape correction')
    parser.add_argument('--no-normalization', action='store_true', help='Disable intensity normalization')
    parser.add_argument('--nb-points', type=int, default=nb_of_points_per_profile, help='Points per profile (default from config)')
    parser.add_argument('--plot', action='store_true', help='Plot predictions')
    parser.add_argument('--batch-size', type=int, default=batch_size, help='Batch size for prediction')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory (relative to data_folder if not absolute)')
    parser.add_argument('--out-name', type=str, default=None, help='Output filename (overrides template)')
    parser.add_argument('--out-path', type=str, default=None, help='Full output file path (highest priority)')
    args = parser.parse_args()

    # Allow CLI overrides of config defaults
    Q_MIN = args.q_min
    Q_MAX = args.q_max
    Convert_to_q = not args.no_convert_to_q
    shape_correction = not args.no_shape_correction
    Normalization = not args.no_normalization
    nb_of_points_per_profile = args.nb_points
    plot_data = args.plot
    batch_size = args.batch_size

    if Convert_to_q and args.wavelength is None:
        raise ValueError("Wavelength is required when converting 2-theta to q. Use -w/--wavelength or pass --no-convert-to-q.")

    main(
        args.data_folder,
        name=args.name,
        start=args.start,
        end=args.end,
        wave_length=args.wavelength,
        out_dir=args.out_dir,
        out_name=args.out_name,
        out_path=args.out_path,
    )
