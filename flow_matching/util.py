# code largely adapted from labs2 of https://diffusion.csail.mit.edu/
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import random

import numpy as np
import torch
import torch.distributions as D

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
import seaborn as sns
from einops import rearrange
from scipy.ndimage import gaussian_filter

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def set_all_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass

class Density(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass

class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)

class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_1D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 1) - 0.5) * scale + x_offset * torch.Tensor([1.0])
        covs = torch.diag_embed(torch.ones(nmodes, 1)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_1D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        means = torch.linspace(-scale, scale, nmodes)[:nmodes].unsqueeze(1) + torch.Tensor([1.0]) * x_offset
        print(means)
        covs = torch.diag_embed(torch.ones(nmodes, 1) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
    
    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

# Several plotting utility functions
def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])

    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)

def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, bins=200, scale: float = 5.0, percentile: int = 99, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu() # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)

def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)

def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    sns.kdeplot(x=samples[:,0].cpu(), y=samples[:,1].cpu(), ax=ax, **kwargs)

def imshow_density(density: Density, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], bins: int, ax: Optional[Axes] = None, x_offset: float = 0.0, **kwargs):
    device = get_device()
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)

def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, x_offset:float = 0.0, **kwargs):
    device = get_device()
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), origin='lower', **kwargs)

def hist1d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, bins: int = 200, range: Optional[Tuple[float, float]] = None, **kwargs):
    assert sampleable.dim == 1
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu() # (ns, 1)
    ax.hist(samples.numpy(), bins=bins, range=range, density=True, **kwargs)

def plot_path(path, ts: torch.Tensor, ax: Optional[Axes] = None):
    if ax is None:
        ax = plt.gca()
    path_num = path.shape[1] # (time, path_num, dim)
    for i in range(path_num):
        y_data = path[:, i, 0]
        ax.plot(ts.cpu(), y_data.cpu(), lw=3, label=f'Path {i+1}')

def plot_path_animation(path, ts: torch.Tensor, fig: plt.Figure, ax: Optional[Axes] = None):
    plt.rcParams['animation.embed_limit'] = 2**128    
    if ax is None:
        ax = plt.gca()
    timesteps, path_num = path.shape[:2] # (time, path_num, dim)
    path = path.cpu()
    ts = ts.cpu()

    line_cmap = cm.get_cmap('Paired', path_num) # Get a colormap with enough distinct colors

    # Initialize lines and separate marker objects
    lines = []
    markers = [] # List to hold our marker objects
    for i in range(path_num):
        # Main line for the path
        line_color = line_cmap(i)
        line_obj, = ax.plot([], [], lw=4, label=f'Path {i+1}', color=line_color)
        lines.append(line_obj)

        # Marker for the current point of this path
        # Use a specific marker style (e.g., 'o' for circle), size, and color
        # The color will be inherited from the line's color if you don't specify,
        # or you can explicitly set it to match the line.
        marker_obj, = ax.plot([], [], 'o', markersize=10, color=line_obj.get_color())
        markers.append(marker_obj)

    # --- Define the initialization function ---
    def init():
        # Set all lines and markers to empty
        for line in lines:
            line.set_data([], [])
        for marker in markers:
            marker.set_data([], [])
        return lines + markers # Return all artists that will be modified

    # --- Define the update function ---
    def update(frame):

        for i, (line, marker) in enumerate(zip(lines, markers)):
            y_data = path[:frame+1, i, 0]
            # Update the line data up to the current frame
            line.set_data(ts[:frame+1], y_data)
            # Update the marker to show only the current point (the last point of the line)
            marker.set_data([ts[frame]], [y_data[frame]])            
        return lines + markers # Return ALL artists that were updated
    # Create the animation
    # frames: The number of frames to animate (equal to timesteps)
    # interval: Delay between frames in milliseconds
    # blit: Optimization to only redraw parts that have changed
    anim = FuncAnimation(
        fig,
        update,
        frames=timesteps,
        init_func=init,
        blit=True,
        interval=50 # 50 ms per frame = 20 frames per second
    )

    return anim

def plot_heatmap(path, ts, ax: Optional[Axes] = None, bins=200, y_min = -15.0, y_max = 15.0):
    if ax is None:
        ax = plt.gca()
    timesteps, path_num = path.shape[:2] # (time, path_num, dim)
    path = path.cpu()
    ts = rearrange(ts, 'n t 1 -> t n')
    all_y_values = path[:,:,0].flatten().cpu().numpy()
    all_x_values = ts.flatten().cpu().numpy()
    H, xedges, yedges = np.histogram2d(
        all_x_values,
        all_y_values,
        bins=[timesteps, bins],
        range=[[0, 1], [y_min, y_max]]
    )
    H = H.T
    # 3. NORMALIZE the Density Matrix Column by Column
    # This is the key step! We normalize the conditional probability P(Y|X)
    # We sum along axis 0 (down the columns) to get the total count for each X-bin.
    column_sums = H.max(axis=0)
    # Divide every element in H by the sum of its column.
    # np.divide handles broadcasting automatically.
    H_normalized = np.divide(H, column_sums)
    H_smoothed = gaussian_filter(H_normalized, sigma=1.0) # sigma controls blur amount

    #H_normalized = H_normalized.T  # Transpose back for correct orientation
    # Plot the heatmap using pcolormesh
    # 'cmap' sets the color scheme (e.g., 'viridis', 'hot', 'Blues')
    # 'alpha' makes it slightly transparent so grid lines/labels are visible
    # 'zorder=0' ensures it stays in the very background
    #original_cmap = plt.cm.Oranges
    #num_colors = 256  # Number of colors in the colormap
    #colors = original_cmap(np.linspace(0, 0.5, num_colors)) # Takes colors from 0 to 0.5 of the original range
    #new_cmap = mcolors.ListedColormap(colors)
    heatmap = ax.pcolormesh(xedges, yedges, H_smoothed, cmap='viridis', alpha=1.0, zorder=0)
    return heatmap, H_normalized