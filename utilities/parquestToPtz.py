import pandas as pd
import numpy as np
from tqdm import tqdm


def parquet_to_npz(parquet_file, npz_file):
    # Load the entire Parquet file
    df = pd.read_parquet(parquet_file)

    # Initialize dictionary to hold the columns' data
    columns_data = {}

    # Process each column, ensuring data is in 2D form
    for column in tqdm(df.columns, desc="Converting to NPZ"):
        # Ensuring each column is a 2D array with one column
        columns_data[column] = df[column].values.reshape(-1, 1)

    # Concatenate all columns to form a 2D array
    combined_data = np.hstack(list(columns_data.values()))

    # Save to NPZ file with a single array
    np.savez(npz_file, data=combined_data)
    print(f"Data saved to {npz_file}")


def main():
    parquet_file = (
        "./data/testing/final_embeddings.parquet"  # Replace with your file path
    )
    npz_file = "./data/testing/finalTesting.npz"
    parquet_to_npz(parquet_file, npz_file)


if __name__ == "__main__":
    main()
