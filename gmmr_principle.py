"""
This script demonstrates the application of the GMR algorithm to generate a
trajectory of a Gaussian mixture model.

The script generates data points following an "S" shape, fits a Gaussian mixture model
to the data, and then uses the GMR algorithm to generate a trajectory of the model.
The resulting trajectory is plotted and saved as a sequence of images, which are
then combined into a GIF.

The script consists of the following steps:
1. Generate data: The script generates data points following an "S" shape using
   the `generate_s_shape_data` function.
2. Fit a GMM: The script fits a Gaussian mixture model to the generated data using
   the `fit_gmm` function.
3. Use GMR: The script uses the GMR algorithm to generate a trajectory of the fitted
   model using the `gmr` function.
4. Create plots: The script creates plots of the data points and the generated
   trajectory using the `create_plots` function. The plots are saved as images.
5. Create GIF: The script combines the saved images into a GIF using the
   `create_gif_from_images` function.

The resulting GIF is saved as `gmm_gmr.gif` in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import imageio
import os
import shutil


def generate_s_shape_data(n_samples=500):
    """
    Generate data points following an "S" shape.

    Parameters
    ----------
    n_samples : int, optional
        Number of data points to generate. Default is 500.

    Returns
    -------
    data : ndarray, shape (n_samples, 2)
        Data points following an "S" shape.
    """
    t = np.linspace(0, 1, n_samples)
    x = np.sin(2 * np.pi * t)
    y = np.sign(t - 0.5) * (1 - np.cos(2 * np.pi * t))
    return np.vstack((x, y)).T


def fit_gmm(data, n_components=6):
    """
    Fit a Gaussian mixture model to the data.

    Parameters
    ----------
    data : ndarray, shape (n_samples, 2)
        Data points.
    n_components : int, optional
        Number of components in the mixture model. Default is 6.

    Returns
    -------
    gmm : GaussianMixture
        Fitted Gaussian mixture model.
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(data)
    return gmm


def gmr(gmm, x_vals):
    """
    Use the GMR algorithm to generate a trajectory of the Gaussian mixture model.

    Parameters
    ----------
    gmm : GaussianMixture
        Fitted Gaussian mixture model.
    x_vals : ndarray, shape (n_samples,)
        Values of the independent variable.

    Returns
    -------
    mean : ndarray, shape (n_samples, 2)
        Mean of the mixture model at each `x_vals`.
    covariance : ndarray, shape (n_samples, 2, 2)
        Covariance of the mixture model at each `x_vals`.
    """
    mean = np.zeros((len(x_vals), 2))
    covariance = np.zeros((len(x_vals), 2, 2))

    for i, x in enumerate(x_vals):
        weights = gmm.predict_proba([[x, 0]])[0]
        mean[i] = np.dot(weights, gmm.means_)
        covariance[i] = np.sum(weights[:, None, None] * gmm.covariances_, axis=0)

    return mean, covariance


def create_plots(data, gmm, means, covariances, i, directory):
    """
    Create plots of the data points and the generated trajectory.

    Parameters
    ----------
    data : ndarray, shape (n_samples, 2)
        Data points.
    gmm : GaussianMixture
        Fitted Gaussian mixture model.
    means : ndarray, shape (n_samples, 2)
        Means of the mixture model at each `x_vals`.
    covariances : ndarray, shape (n_samples, 2, 2)
        Covariances of the mixture model at each `x_vals`.
    i : int
        Index of the current frame.
    directory : str
        Directory to save the images.
    """
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=4, label='Data points')
    ax.plot(means[:, 0], means[:, 1], color='red', label='GMR Trajectory')

    for j in range(means.shape[0]):
        cov = covariances[j]
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi

        ell = plt.matplotlib.patches.Ellipse(means[j], v[0], v[1], 180.0 + angle, color='red', alpha=0.1)
        ax.add_artist(ell)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    image_filename = os.path.join(directory, f'frame_{i}.png')
    fig.savefig(image_filename)
    plt.close(fig)


def create_gif_from_images(directory, output_filename='gmm_gmr.gif'):
    """
    Create a GIF from saved images.

    Parameters
    ----------
    directory : str
        Directory containing the saved images.
    output_filename : str, optional
        Name of the output GIF file. Default is 'gmm_gmr.gif'.
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(directory, filename)))
    imageio.mimsave(output_filename, images, duration=0.2)


# Main code
data = generate_s_shape_data()
gmm = fit_gmm(data)

# Temporary directory for images
temp_dir = 'temp_frames'
os.makedirs(temp_dir, exist_ok=True)

# Generate frames
x_vals = np.linspace(-1, 1, 20)
means, covariances = gmr(gmm, x_vals)

for i in range(1, len(x_vals) + 1):
    create_plots(data, gmm, means[:i], covariances[:i], i, temp_dir)

# Create GIF from saved images
create_gif_from_images(temp_dir)

# Clean up temporary files
shutil.rmtree(temp_dir)

