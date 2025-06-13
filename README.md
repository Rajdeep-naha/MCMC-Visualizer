# MCMC Visualizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mcmc-visualizer.streamlit.app/)

A Streamlit-based web application that provides interactive visualizations of three fundamental Markov Chain Monte Carlo (MCMC) methods: Metropolis-Hastings, Gibbs Sampling, and Simulated Annealing. The application allows users to explore how these algorithms sample from different probability distributions with real-time visualization.

🔗 **Live Demo**: [https://mcmc-visualizer.streamlit.app/](https://mcmc-visualizer.streamlit.app/)

## Features

- **Interactive Visualizations**: Real-time animations of MCMC sampling processes
- **Multiple Algorithms**:
  - Metropolis-Hastings
  - Gibbs Sampling
  - Simulated Annealing
- **Target Distributions**:
  - Banana-shaped distribution
  - Bivariate Gaussian distribution
- **Customizable Parameters**: Adjust algorithm parameters and distribution settings
- **Side-by-Side Comparison**: Compare how different algorithms explore the same probability space
- **Performance Metrics**: Track acceptance rates and convergence metrics

## Live Demo

You can try out the application without any installation by visiting the live demo:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mcmc-visualizer.streamlit.app/)

Or visit: https://mcmc-visualizer.streamlit.app/

The demo allows you to:
- Compare different MCMC algorithms in real-time
- Adjust parameters and see immediate effects
- Visualize the sampling process with interactive plots
- No installation or setup required!

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajdeep-naha/MCMC-Visualizer.git
   cd MCMC-Visualizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser. If it doesn't, navigate to `http://localhost:8501`

3. Use the sidebar to:
   - Select the target distribution
   - Adjust distribution parameters
   - Set the number of iterations
   - Configure algorithm-specific parameters

## Understanding the Visualization

- **Blue dots**: Trail of Metropolis-Hastings algorithm samples
- **Orange dots**: Trail of Gibbs Sampling algorithm samples
- **Green dots**: Trail of Simulated Annealing algorithm samples
- **Red dot**: Current position of the sampler
- **Circle markers**: Accepted moves
- **Cross markers (red)**: Rejected moves
- **Background contour**: Target probability distribution

## Project Structure

- `app.py`: Main Streamlit application
- `utils/`: Contains the core functionality
  - `distributions.py`: Implementation of probability distributions
  - `mcmc_algorithms.py`: MCMC algorithm implementations
  - `visualization.py`: Visualization utilities
- `requirements.txt`: Python dependencies

## Dependencies

- Python 3.7+
- Streamlit
- NumPy
- Plotly
- SciPy
- Pandas
- Pillow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here's how you can contribute:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes and test them
4. Submit a pull request with a clear description of your changes

### Running Tests

Before submitting a PR, please make sure all tests pass:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest
```

### Reporting Issues

If you find any bugs or have suggestions, please open an issue on GitHub.

## License

This project is open source and available under the MIT License.

## Author

Rajdeep Naha - [GitHub Profile](https://github.com/Rajdeep-naha)
