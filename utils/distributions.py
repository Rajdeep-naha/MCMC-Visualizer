import numpy as np
from scipy import stats

class Distribution:
    """Base class for probability distributions."""
    
    def __init__(self):
        pass
    
    def pdf(self, x):
        """Probability density function."""
        raise NotImplementedError
    
    def log_pdf(self, x):
        """Log probability density function."""
        return np.log(self.pdf(x))

class BananaDistribution(Distribution):
    """
    Banana-shaped distribution for testing MCMC methods.
    Based on a transformed bivariate Gaussian.
    
    Parameters:
    -----------
    a : float
        Controls the curvature of the banana shape
    b : float
        Controls the width of the banana
    """
    
    def __init__(self, a=0.5, b=4.0):
        super().__init__()
        self.a = a
        self.b = b
    
    def pdf(self, x):
        """
        Probability density function of the banana distribution.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Point where to evaluate the PDF
        
        Returns:
        --------
        float
            PDF value at x
        """
        # Transform x to get banana shape
        x1 = x[0]
        x2 = x[1] - self.a * x[0]**2 - self.b
        
        # Use a bivariate Gaussian with zero mean and unit variance
        return stats.multivariate_normal.pdf([x1, x2], mean=[0, 0], cov=np.eye(2))
    
    def conditional_pdf(self, x, dim, value):
        """
        Conditional PDF for a given dimension with the other dimension fixed.
        Used for Gibbs sampling.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Current state
        dim : int
            Dimension to condition on (0 or 1)
        value : float
            Value of the other dimension
        
        Returns:
        --------
        function
            Conditional PDF function that takes a single argument
        """
        if dim == 0:
            # p(x0 | x1 = value)
            def conditional(x0):
                return self.pdf([x0, value])
        else:
            # p(x1 | x0 = value)
            def conditional(x1):
                return self.pdf([value, x1])
        
        return conditional

class BivariateGaussianDistribution(Distribution):
    """
    Bivariate Gaussian distribution.
    
    Parameters:
    -----------
    mean : array-like, shape (2,)
        Mean vector
    cov : array-like, shape (2, 2)
        Covariance matrix
    """
    
    def __init__(self, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]):
        super().__init__()
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.precision = np.linalg.inv(self.cov)
        self.normalizing_const = (
            (2 * np.pi) ** (-1) * 
            np.linalg.det(self.cov) ** (-0.5)
        )
    
    def pdf(self, x):
        """
        Probability density function of the bivariate Gaussian.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Point where to evaluate the PDF
        
        Returns:
        --------
        float
            PDF value at x
        """
        x = np.array(x)
        diff = x - self.mean
        exponent = -0.5 * diff.T @ self.precision @ diff
        return self.normalizing_const * np.exp(exponent)
    
    def conditional_pdf(self, x, dim, value):
        """
        Conditional PDF for a given dimension with the other dimension fixed.
        Used for Gibbs sampling.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Current state
        dim : int
            Dimension to condition on (0 or 1)
        value : float
            Value of the other dimension
        
        Returns:
        --------
        function
            Conditional PDF function that takes a single argument
        """
        if dim == 0:
            # p(x0 | x1 = value)
            def conditional(x0):
                return self.pdf([x0, value])
        else:
            # p(x1 | x0 = value)
            def conditional(x1):
                return self.pdf([value, x1])
        
        return conditional
    
    def conditional_sample(self, x, dim):
        """
        Sample from the conditional distribution.
        Used for Gibbs sampling.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Current state
        dim : int
            Dimension to sample (0 or 1)
        
        Returns:
        --------
        float
            Sample from the conditional distribution
        """
        other_dim = 1 - dim
        
        # Extract relevant parameters
        mu = self.mean
        sigma = np.sqrt(np.diag(self.cov))
        rho = self.cov[0, 1] / (sigma[0] * sigma[1])
        
        # Compute conditional mean and variance
        cond_mean = mu[dim] + rho * (sigma[dim] / sigma[other_dim]) * (x[other_dim] - mu[other_dim])
        cond_var = (1 - rho**2) * self.cov[dim, dim]
        
        # Sample from conditional normal distribution
        return np.random.normal(cond_mean, np.sqrt(cond_var))
