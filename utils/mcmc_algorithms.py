import numpy as np
from scipy import stats

class MCMC:
    """Base class for MCMC algorithms."""
    
    def __init__(self, distribution, step_size=0.1):
        """
        Initialize the MCMC sampler.
        
        Parameters:
        -----------
        distribution : Distribution
            Target distribution to sample from
        step_size : float
            Size of the proposal step
        """
        self.distribution = distribution
        self.step_size = step_size
    
    def propose(self, current_state):
        """
        Propose a new state given the current state.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        
        Returns:
        --------
        array-like
            Proposed next state
        """
        raise NotImplementedError
    
    def accept(self, current_state, proposed_state):
        """
        Determine whether to accept the proposed state.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        proposed_state : array-like
            Proposed next state
        
        Returns:
        --------
        bool
            Whether to accept the proposed state
        """
        raise NotImplementedError
    
    def run(self, initial_state, n_iterations):
        """
        Run the MCMC sampler.
        
        Parameters:
        -----------
        initial_state : array-like
            Initial state of the Markov chain
        n_iterations : int
            Number of iterations to run
        
        Returns:
        --------
        array-like, shape (n_iterations + 1, d)
            Samples from the chain, including the initial state
        array-like, shape (n_iterations,)
            Boolean array indicating whether each proposal was accepted
        """
        d = len(initial_state)
        samples = np.zeros((n_iterations + 1, d))
        accepts = np.zeros(n_iterations, dtype=bool)
        
        samples[0] = initial_state
        
        for i in range(n_iterations):
            proposed_state = self.propose(samples[i])
            accepts[i] = self.accept(samples[i], proposed_state)
            
            if accepts[i]:
                samples[i + 1] = proposed_state
            else:
                samples[i + 1] = samples[i]
        
        return samples, accepts

class MetropolisHastings(MCMC):
    """
    Metropolis-Hastings MCMC algorithm.
    
    Uses symmetric proposal distribution (random walk).
    """
    
    def propose(self, current_state):
        """
        Propose a new state using a random walk.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        
        Returns:
        --------
        array-like
            Proposed next state
        """
        return current_state + self.step_size * np.random.randn(len(current_state))
    
    def accept(self, current_state, proposed_state):
        """
        Determine whether to accept the proposed state using the Metropolis-Hastings criterion.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        proposed_state : array-like
            Proposed next state
        
        Returns:
        --------
        bool
            Whether to accept the proposed state
        """
        # Compute log PDF for current and proposed states
        current_log_pdf = self.distribution.log_pdf(current_state)
        proposed_log_pdf = self.distribution.log_pdf(proposed_state)
        
        # Compute acceptance probability
        log_alpha = proposed_log_pdf - current_log_pdf
        
        # Accept or reject
        return np.log(np.random.rand()) < min(0, log_alpha)

class GibbsSampling(MCMC):
    """
    Gibbs Sampling MCMC algorithm.
    
    Samples each dimension conditionally.
    """
    
    def propose(self, current_state):
        """
        Propose a new state by sampling each dimension conditionally.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        
        Returns:
        --------
        array-like
            Proposed next state
        """
        proposed_state = current_state.copy()
        d = len(current_state)
        
        # Sample a random dimension to update
        dim = np.random.randint(d)
        
        # If distribution has an analytical form for conditional sampling
        if hasattr(self.distribution, 'conditional_sample'):
            proposed_state[dim] = self.distribution.conditional_sample(current_state, dim)
        else:
            # Otherwise use numerical approximation
            # Create grid of points for numerical sampling
            other_dim = 1 - dim  # For 2D case
            grid_size = 1000
            grid_min = current_state[dim] - 5 * self.step_size
            grid_max = current_state[dim] + 5 * self.step_size
            grid = np.linspace(grid_min, grid_max, grid_size)
            
            # Evaluate PDF at each grid point
            pdf_values = np.zeros(grid_size)
            for i, g in enumerate(grid):
                point = current_state.copy()
                point[dim] = g
                pdf_values[i] = self.distribution.pdf(point)
            
            # Normalize to get PMF
            pmf = pdf_values / np.sum(pdf_values)
            
            # Sample from discrete distribution
            idx = np.random.choice(grid_size, p=pmf)
            proposed_state[dim] = grid[idx]
        
        return proposed_state
    
    def accept(self, current_state, proposed_state):
        """
        In Gibbs sampling, we always accept the proposed state.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        proposed_state : array-like
            Proposed next state
        
        Returns:
        --------
        bool
            Always True for Gibbs sampling
        """
        return True

class SimulatedAnnealing(MCMC):
    """
    Simulated Annealing algorithm.
    
    Uses a temperature parameter that decreases over time.
    """
    
    def __init__(self, distribution, step_size=0.1, initial_temp=5.0, cooling_rate=0.95):
        """
        Initialize the Simulated Annealing sampler.
        
        Parameters:
        -----------
        distribution : Distribution
            Target distribution to sample from
        step_size : float
            Size of the proposal step
        initial_temp : float
            Initial temperature
        cooling_rate : float
            Rate at which temperature decreases (multiplier)
        """
        super().__init__(distribution, step_size)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.current_temp = initial_temp
    
    def propose(self, current_state):
        """
        Propose a new state using a random walk.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        
        Returns:
        --------
        array-like
            Proposed next state
        """
        return current_state + self.step_size * np.random.randn(len(current_state))
    
    def accept(self, current_state, proposed_state):
        """
        Determine whether to accept the proposed state using the Metropolis criterion
        modified with the temperature parameter.
        
        Parameters:
        -----------
        current_state : array-like
            Current state of the Markov chain
        proposed_state : array-like
            Proposed next state
        
        Returns:
        --------
        bool
            Whether to accept the proposed state
        """
        # Compute log PDF for current and proposed states
        current_log_pdf = self.distribution.log_pdf(current_state)
        proposed_log_pdf = self.distribution.log_pdf(proposed_state)
        
        # Compute acceptance probability with temperature
        log_alpha = (proposed_log_pdf - current_log_pdf) / self.current_temp
        
        # Accept or reject
        return np.log(np.random.rand()) < min(0, log_alpha)
    
    def run(self, initial_state, n_iterations):
        """
        Run the Simulated Annealing sampler.
        
        Parameters:
        -----------
        initial_state : array-like
            Initial state of the Markov chain
        n_iterations : int
            Number of iterations to run
        
        Returns:
        --------
        array-like, shape (n_iterations + 1, d)
            Samples from the chain, including the initial state
        array-like, shape (n_iterations,)
            Boolean array indicating whether each proposal was accepted
        array-like, shape (n_iterations + 1,)
            Temperature at each iteration
        """
        d = len(initial_state)
        samples = np.zeros((n_iterations + 1, d))
        accepts = np.zeros(n_iterations, dtype=bool)
        temperatures = np.zeros(n_iterations + 1)
        
        samples[0] = initial_state
        temperatures[0] = self.initial_temp
        self.current_temp = self.initial_temp
        
        for i in range(n_iterations):
            proposed_state = self.propose(samples[i])
            accepts[i] = self.accept(samples[i], proposed_state)
            
            if accepts[i]:
                samples[i + 1] = proposed_state
            else:
                samples[i + 1] = samples[i]
            
            # Cool down temperature
            self.current_temp *= self.cooling_rate
            temperatures[i + 1] = self.current_temp
        
        return samples, accepts, temperatures
