import numpy as np


def split_npz(npz_file, split_index, output_file1, output_file2):
    # Load the data from the npz file
    data = np.load(npz_file)
    keys = data.files  # Get the list of keys
    print(f"Keys in the NPZ file: {keys}")

    # Assuming there's only one array to split, or specify the key if known
    key_to_split = keys[0]  # Use the first key, or specify your key
    combined_data = data[key_to_split]

    # Ensure that the split index is within the bounds
    if split_index >= combined_data.shape[0] or split_index < 0:
        raise ValueError(f"Split index {split_index} is out of bounds.")

    # Split the data at the specified index
    data_part1 = combined_data[:split_index, :]
    data_part2 = combined_data[split_index:, :]

    # Save the split data to new npz files
    np.savez(output_file1, data=data_part1)
    np.savez(output_file2, data=data_part2)

    print(f"Data split into {output_file1} and {output_file2}")
    print(f"Shape of part 1: {data_part1.shape}")
    print(f"Shape of part 2: {data_part2.shape}")


def main():
    npz_file = "./data/combined/reduced_data.npz"  # Replace with your actual file path
    output_file1 = "./data/split/part1.npz"
    output_file2 = "./data/split/part2.npz"
    split_index = 465328  # The index to split at

    split_npz(npz_file, split_index, output_file1, output_file2)


if __name__ == "__main__":
    main()
