import numpy as np


def combine_npz_files(npz_file1, npz_file2, output_file):
    # Load data from the first npz file
    data1 = np.load(npz_file1)
    data_dict1 = {key: data1[key] for key in data1.files}
    data1.close()

    # Load data from the second npz file
    data2 = np.load(npz_file2)
    data_dict2 = {key: data2[key] for key in data2.files}
    data2.close()

    # Merge the dictionaries by concatenating arrays with the same key
    combined_data = {}
    for key in data_dict1:
        if key in data_dict2:
            # Concatenate the arrays for the same key
            combined_data[key] = np.concatenate((data_dict1[key], data_dict2[key]))
        else:
            # Keep the original array if the key is not in data_dict2
            combined_data[key] = data_dict1[key]

    # Include any keys only in data_dict2
    for key in data_dict2:
        if key not in combined_data:
            combined_data[key] = data_dict2[key]

    # Save the combined data to a new npz file
    np.savez(output_file, **combined_data)
    print(f"Combined data saved to {output_file}")


def main():
    npz_file1 = "./data/testing/finalTesting.npz"
    npz_file2 = "./data/training/finalTraining.npz"
    output_file = "./data/combined/combinedData.npz"

    combine_npz_files(npz_file1, npz_file2, output_file)


if __name__ == "__main__":
    main()
