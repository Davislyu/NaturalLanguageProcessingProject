import numpy as np
import pandas as pd


def load_labels(csv_file):
    """Load labels from the CSV file and return them as a numpy array."""
    labels_df = pd.read_csv(csv_file)
    labels = labels_df["LABEL"].values  # Adjust column name if necessary
    return labels


def add_labels_to_npz(npz_file, labels, output_file):
    """Add labels as a new column to the NPZ data and save to a new NPZ file."""
    # Load data from the NPZ file
    data = np.load(npz_file)
    key = data.files[0]  # Assuming the data is under the first key
    combined_data = data[key]

    # Ensure labels match the number of rows
    if len(labels) != combined_data.shape[0]:
        raise ValueError(
            "The number of labels does not match the number of rows in the NPZ data."
        )

    # Add labels as a new column
    labeled_data = np.column_stack((combined_data, labels))

    # Save the updated data
    np.savez(output_file, data=labeled_data)
    print(f"Labeled data saved to {output_file}")
    print(f"Shape of data before: {combined_data.shape}")
    print(f"Shape of data after: {labeled_data.shape}")


def main():
    npz_file = "./data/split/Pca_traningSet.npz"  # File with data to label
    csv_file = "./data/training/trainingSetRaw.csv"  # CSV file with labels
    output_file = "./data/split/Pca_traningSet_labeled.npz"  # Output file with labels

    # Load labels from the CSV file
    labels = load_labels(csv_file)

    # Add labels to the NPZ file and save
    add_labels_to_npz(npz_file, labels, output_file)


if __name__ == "__main__":
    main()
