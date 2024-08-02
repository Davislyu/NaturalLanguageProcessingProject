import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
from tqdm import tqdm


def load_npz_data(npz_file):
    """Load and combine data from all arrays in the NPZ file."""
    data = np.load(npz_file)
    combined_data = []

    print("Checking data structure...")
    first_shape = None
    for key in data.files:
        array = data[key]
        print(f"Array name: {key}, shape: {array.shape}")
        if first_shape is None:
            first_shape = array.shape
        if array.shape[0] != first_shape[0]:
            raise ValueError("All arrays must have the same number of rows.")
        combined_data.append(array)

    combined_data = np.hstack(combined_data)  # Stack arrays horizontally
    return combined_data


def perform_pca(X):
    """Perform PCA and return the cumulative variance explained."""
    print("Performing PCA...")
    pca = PCA().fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return cumulative_variance


def find_optimal_components(cumulative_variance):
    """Find the optimal number of components using the elbow method."""
    print("Determining the optimal number of components using the elbow method...")
    kneedle = KneeLocator(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        curve="concave",
        direction="increasing",
        S=1.0,
        interp_method="polynomial",
    )
    optimal_components = kneedle.elbow
    return optimal_components, kneedle


def plot_variance_curve(cumulative_variance, optimal_components):
    """Plot the explained variance curve and the identified elbow point."""
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker="o", linestyle="-", color="b")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Components")

    if optimal_components is not None:
        plt.axvline(
            optimal_components,
            linestyle="--",
            color="r",
            label=f"Optimal components: {optimal_components}",
        )
    plt.legend()
    plt.grid()
    plt.show()


def main():
    npz_file = "./data/combined/combinedData.npz"  # Replace with your actual file path
    try:
        data = load_npz_data(npz_file)
        print(f"Data shape before PCA: {data.shape}")
        cumulative_variance = perform_pca(data)
        optimal_components, _ = find_optimal_components(cumulative_variance)
        plot_variance_curve(cumulative_variance, optimal_components)

        if optimal_components is not None:
            print(
                f"The optimal number of components identified is: {optimal_components}"
            )
            user_input = (
                input(
                    "Do you want to proceed with this number of components for PCA? (yes/no): "
                )
                .strip()
                .lower()
            )
            if user_input == "yes":
                print(f"Proceeding with PCA using {optimal_components} components...")
                # Proceed with PCA using the optimal number of components
                pca = PCA(n_components=optimal_components)
                reduced_data = pca.fit_transform(data)
                # Save the reduced data
                np.savez("reduced_data.npz", reduced_data=reduced_data)
                print(
                    "PCA completed and data saved to './data/combined/reduced_data.npz'."
                )
                print(f"Data shape after PCA: {reduced_data.shape}")
            else:
                print("PCA process aborted by the user.")
        else:
            print("Failed to identify the optimal number of components.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
