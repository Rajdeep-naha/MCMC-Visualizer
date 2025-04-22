import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.distributions import BananaDistribution, BivariateGaussianDistribution
from utils.mcmc_algorithms import MetropolisHastings, GibbsSampling, SimulatedAnnealing
from utils.visualization import create_contour_plot, create_animation_frames, add_annotations

# Set page configuration
st.set_page_config(
    page_title="MCMC Visualization",
    page_icon="📊",
    layout="wide",
)

# App title and description
st.title("Markov Chain Monte Carlo (MCMC) Methods Visualization")
st.markdown("""
    This application visualizes and compares three Markov Chain Monte Carlo methods:
    **Metropolis-Hastings**, **Gibbs Sampling**, and **Simulated Annealing**.
    Each method is demonstrated on the same target distribution, showing how they explore
    the probability space differently.
""")

# Add color explanation at the beginning
st.markdown("""
    ### Understanding the Visualization
    
    In the animations below:
    - **Blue dots** (left panel): Trail of Metropolis-Hastings algorithm samples
    - **Orange dots** (middle panel): Trail of Gibbs Sampling algorithm samples
    - **Green dots** (right panel): Trail of Simulated Annealing algorithm samples
    - **Red dot**: Current position of the sampler at this iteration
    - **Circle markers**: Accepted moves (where the proposal was accepted)
    - **Cross markers (red)**: Rejected moves (where the algorithm stayed at the same position)
    
    The background contour shows the target probability distribution being sampled.
""")

# Sidebar for controls
st.sidebar.header("Parameters")

# Select distribution
distribution_type = st.sidebar.selectbox(
    "Target Distribution", ["Banana-shaped", "Bivariate Gaussian"])

# Distribution parameters
if distribution_type == "Banana-shaped":
    banana_a = st.sidebar.slider("Banana curvature (a)", 0.1, 2.0, 0.5)
    banana_b = st.sidebar.slider("Banana width (b)", 1.0, 10.0, 4.0)
    distribution = BananaDistribution(a=banana_a, b=banana_b)
    x_range = (-4, 4)
    y_range = (-2, 14)
else:  # Bivariate Gaussian
    mean_x = st.sidebar.slider("Mean X", -2.0, 2.0, 0.0)
    mean_y = st.sidebar.slider("Mean Y", -2.0, 2.0, 0.0)
    sigma_x = st.sidebar.slider("Sigma X", 0.1, 2.0, 1.0)
    sigma_y = st.sidebar.slider("Sigma Y", 0.1, 2.0, 1.0)
    corr = st.sidebar.slider("Correlation", -0.99, 0.99, 0.5)
    distribution = BivariateGaussianDistribution(
        mean=[mean_x, mean_y],
        cov=[[sigma_x**2, corr * sigma_x * sigma_y],
             [corr * sigma_x * sigma_y, sigma_y**2]])
    x_range = (mean_x - 4 * sigma_x, mean_x + 4 * sigma_x)
    y_range = (mean_y - 4 * sigma_y, mean_y + 4 * sigma_y)

# MCMC parameters
st.sidebar.header("MCMC Settings")
n_iterations = st.sidebar.slider("Number of Iterations", 100, 2000, 500)
step_size = st.sidebar.slider("Step Size", 0.01, 2.0, 0.3)

# Specific parameters for Simulated Annealing
initial_temp = st.sidebar.slider("Initial Temperature (SA)", 1.0, 10.0, 5.0)
cooling_rate = st.sidebar.slider("Cooling Rate (SA)", 0.8, 0.99, 0.95)

# Animation controls
st.sidebar.header("Animation")
animation_speed = st.sidebar.slider("Animation Speed", 10, 200, 50, 10)
trail_length = st.sidebar.slider("Trail Length", 1, 50, 15)

# Initialize algorithms
initial_state = np.array([0.0, 0.0])
mh = MetropolisHastings(distribution, step_size)
gibbs = GibbsSampling(distribution, step_size)
sa = SimulatedAnnealing(distribution, step_size, initial_temp, cooling_rate)

# Run button to start the simulation
if st.button("Run Simulation"):
    # Set up layout for progress indicators
    progress_container = st.container()
    with progress_container:
        st.markdown("### Simulation Progress")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Metropolis-Hastings**")
            mh_progress = st.progress(0)
            mh_status = st.empty()

        with col2:
            st.markdown("**Gibbs Sampling**")
            gibbs_progress = st.progress(0)
            gibbs_status = st.empty()

        with col3:
            st.markdown("**Simulated Annealing**")
            sa_progress = st.progress(0)
            sa_status = st.empty()

    visualization_status = st.empty()

    # Define modified run function with progress updates
    def run_with_progress(sampler, initial_state, n_iterations, progress_bar,
                          status):
        samples = np.zeros((n_iterations + 1, len(initial_state)))
        accepts = np.zeros(n_iterations, dtype=bool)
        proposals = np.zeros(
            (n_iterations, len(initial_state)))  # Store proposed states
        temperatures = np.zeros(n_iterations + 1) if hasattr(
            sampler, 'current_temp') else None

        samples[0] = initial_state
        if hasattr(sampler, 'current_temp'):
            temperatures[0] = sampler.initial_temp
            sampler.current_temp = sampler.initial_temp

        # Update progress every 5% of iterations
        update_freq = max(1, n_iterations // 20)

        for i in range(n_iterations):
            proposed_state = sampler.propose(samples[i])
            proposals[i] = proposed_state  # Store the proposed state
            accepts[i] = sampler.accept(samples[i], proposed_state)

            if accepts[i]:
                samples[i + 1] = proposed_state
            else:
                samples[i + 1] = samples[i]

            if hasattr(sampler, 'current_temp'):
                sampler.current_temp *= sampler.cooling_rate
                temperatures[i + 1] = sampler.current_temp

            # Update progress bar
            if i % update_freq == 0 or i == n_iterations - 1:
                progress = int(100 * (i + 1) / n_iterations)
                progress_bar.progress(progress)
                status.text(f"Step {i+1}/{n_iterations} ({progress}%)")

        progress_bar.progress(100)
        status.text("Completed")

        if temperatures is not None:
            return samples, accepts, proposals, temperatures
        else:
            return samples, accepts, proposals

    # Run the algorithms with progress updates
    mh_status.text("Starting...")
    mh_result = run_with_progress(mh, initial_state, n_iterations, mh_progress,
                                  mh_status)
    mh_samples, mh_accepts, mh_proposals = mh_result[:3]

    gibbs_status.text("Starting...")
    gibbs_result = run_with_progress(gibbs, initial_state, n_iterations,
                                     gibbs_progress, gibbs_status)
    gibbs_samples, gibbs_accepts, gibbs_proposals = gibbs_result[:3]

    sa_status.text("Starting...")
    sa_result = run_with_progress(sa, initial_state, n_iterations, sa_progress,
                                  sa_status)
    sa_samples, sa_accepts, sa_proposals, sa_temps = sa_result

    visualization_status.info("Generating visualizations...")

    # Calculate acceptance rates
    mh_acceptance_rate = np.mean(mh_accepts) * 100
    gibbs_acceptance_rate = np.mean(gibbs_accepts) * 100
    sa_acceptance_rate = np.mean(sa_accepts) * 100

    # Create contour plots of the target distribution
    x_grid, y_grid, z_values = create_contour_plot(distribution, x_range,
                                                   y_range)

    # Create subplots with 1 row and 3 columns
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Metropolis-Hastings (Accept: {mh_acceptance_rate:.1f}%)",
            f"Simulated Annealing (Accept: {sa_acceptance_rate:.1f}%)",
            f"Gibbs Sampling (Accept: {gibbs_acceptance_rate:.1f}%)"),
        specs=[[{
            'type': 'scatter'
        }, {
            'type': 'scatter'
        }, {
            'type': 'scatter'
        }]])

    # Add contours for each subplot
    for col in range(1, 4):
        fig.add_trace(go.Contour(z=z_values,
                                 x=x_grid[0, :],
                                 y=y_grid[:, 0],
                                 colorscale='Viridis',
                                 opacity=0.6,
                                 showscale=False,
                                 contours=dict(
                                     showlabels=False,
                                     coloring='fill',
                                 )),
                      row=1,
                      col=col)

    # Create animation frames for each method
    frames = create_animation_frames(fig, mh_samples, gibbs_samples,
                                     sa_samples, mh_proposals, gibbs_proposals,
                                     sa_proposals, mh_accepts, gibbs_accepts,
                                     sa_accepts, trail_length,
                                     x_grid, y_grid, z_values)

    # Determine axis ranges to ensure all points are visible
    # For each method, find the min/max values and add some padding
    all_samples = np.vstack([mh_samples, gibbs_samples, sa_samples])
    min_x = np.min(all_samples[:, 0]) - 1
    max_x = np.max(all_samples[:, 0]) + 1
    min_y = np.min(all_samples[:, 1]) - 1
    max_y = np.max(all_samples[:, 1]) + 1

    # Make sure we're not zooming in too much on the distribution
    min_x = min(min_x, x_range[0])
    max_x = max(max_x, x_range[1])
    min_y = min(min_y, y_range[0])
    max_y = max(max_y, y_range[1])

    # Update layout for animation with adjusted axis ranges
    fig.update_layout(title="MCMC Methods Comparison",
                      xaxis_title="X",
                      yaxis_title="Y",
                      height=600,
                      width=1000,
                      updatemenus=[{
                          'type':
                          'buttons',
                          'showactive':
                          False,
                          'buttons': [{
                              'label':
                              'Play',
                              'method':
                              'animate',
                              'args': [
                                  None, {
                                      'frame': {
                                          'duration': animation_speed,
                                          'redraw': True
                                      },
                                      'fromcurrent': True,
                                      'mode': 'immediate',
                                  }
                              ]
                          }, {
                              'label':
                              'Pause',
                              'method':
                              'animate',
                              'args': [[None], {
                                  'frame': {
                                      'duration': 0,
                                      'redraw': False
                                  },
                                  'mode': 'immediate',
                                  'transition': {
                                      'duration': 0
                                  }
                              }]
                          }],
                          'direction':
                          'left',
                          'pad': {
                              'r': 10,
                              't': 87
                          },
                          'x':
                          0.1,
                          'y':
                          0
                      }],
                      sliders=[{
                          'active':
                          0,
                          'yanchor':
                          'top',
                          'xanchor':
                          'left',
                          'currentvalue': {
                              'font': {
                                  'size': 16
                              },
                              'prefix': 'Iteration:',
                              'visible': True,
                              'xanchor': 'right'
                          },
                          'transition': {
                              'duration': 300,
                              'easing': 'cubic-in-out'
                          },
                          'pad': {
                              'b': 10,
                              't': 50
                          },
                          'len':
                          0.9,
                          'x':
                          0.1,
                          'y':
                          0,
                          'steps': [{
                              'args': [[f.name], {
                                  'frame': {
                                      'duration': animation_speed,
                                      'redraw': True
                                  },
                                  'mode': 'immediate',
                                  'transition': {
                                      'duration': animation_speed
                                  }
                              }],
                              'label':
                              str(k),
                              'method':
                              'animate'
                          } for k, f in enumerate(frames)]
                      }])

    # Update axis ranges for all subplots to ensure all points are visible
    fig.update_xaxes(range=[min_x, max_x], row=1, col=1)
    fig.update_yaxes(range=[min_y, max_y], row=1, col=1)
    fig.update_xaxes(range=[min_x, max_x], row=1, col=2)
    fig.update_yaxes(range=[min_y, max_y], row=1, col=2)
    fig.update_xaxes(range=[min_x, max_x], row=1, col=3)
    fig.update_yaxes(range=[min_y, max_y], row=1, col=3)

    # Add annotations
    fig = add_annotations(fig)

    # Show the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display statistics and explanations
    st.header("MCMC Methods Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Metropolis-Hastings")
        st.metric("Acceptance Rate", f"{mh_acceptance_rate:.1f}%")
        st.markdown("""
        **Key Characteristics:**
        - Proposes random moves in any direction
        - Accepts or rejects based on target density ratio
        - Balances exploration and exploitation
        - Simple to implement but can be inefficient in high dimensions
        """)

    with col2:
        st.subheader("Gibbs Sampling")
        st.metric("Acceptance Rate", f"{gibbs_acceptance_rate:.1f}%")
        st.markdown("""
        **Key Characteristics:**
        - Samples one dimension at a time, holding others constant
        - Always accepts moves (acceptance rate = 100% by design)
        - Efficient when conditional distributions are easy to sample
        - Can get stuck in narrow probability regions
        """)

    with col3:
        st.subheader("Simulated Annealing")
        st.metric("Acceptance Rate", f"{sa_acceptance_rate:.1f}%")
        st.markdown("""
        **Key Characteristics:**
        - Uses decreasing temperature parameter
        - Accepts uphill moves more often initially
        - Gradually focuses on exploiting high-probability regions
        - Good for finding global optima in multimodal distributions
        """)

    # Additional educational content
    st.header("Educational Notes")
    st.markdown("""
    ### Understanding Convergence
    
    **Convergence** in MCMC methods refers to how quickly the chain reaches its stationary distribution.
    
    - **Burn-in period**: The initial iterations where the chain hasn't yet converged to the target distribution.
    - **Mixing**: How efficiently the chain explores the entire target distribution.
    
    ### Efficiency Comparison
    
    - **Metropolis-Hastings** is general-purpose but may have lower acceptance rates in complex distributions.
    - **Gibbs Sampling** is highly efficient when variables are correlated and conditional distributions are simple.
    - **Simulated Annealing** is technically not a pure MCMC method as its target distribution changes over time,
      but it's excellent for finding global maxima.
    
    ### Applications
    
    - **Bayesian Statistics**: Parameter estimation, model selection
    - **Statistical Physics**: Simulating physical systems
    - **Machine Learning**: Training neural networks, clustering
    - **Optimization**: Finding global optima in complex functions
    """)

    visualization_status.empty()
else:
    st.info("Set your desired parameters and click 'Run Simulation' to start.")
