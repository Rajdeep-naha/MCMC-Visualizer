import numpy as np
import plotly.graph_objects as go


def create_contour_plot(distribution, x_range, y_range, resolution=100):
    """
    Create a contour plot for a 2D distribution.

    Parameters:
    -----------
    distribution : Distribution
        Distribution to visualize
    x_range : tuple
        Range of x-values (min, max)
    y_range : tuple
        Range of y-values (min, max)
    resolution : int
        Number of points in each dimension

    Returns:
    --------
    tuple
        (x_grid, y_grid, z_values) for creating a contour plot
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    z_values = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            z_values[i, j] = distribution.pdf([x_grid[i, j], y_grid[i, j]])

    return x_grid, y_grid, z_values


def create_animation_frames(fig, mh_samples, gibbs_samples, sa_samples,
                            mh_proposals, gibbs_proposals, sa_proposals,
                            mh_accepts, gibbs_accepts, sa_accepts,
                            trail_length, x_grid, y_grid, z_values):
    """
    Create animation frames for the MCMC visualization.
    """
    frames = []
    n_iterations = len(mh_accepts)

    # Create frames for animation
    for i in range(1, n_iterations + 1):
        frame_data = []

        # Calculate trail start point
        trail_start = max(0, i - trail_length)

        # Add trail data for Gibbs Sampling (middle)
        colors = []
        sizes = []
        symbols = []

        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i + 1):  # Include current point in trail
            if j == i:  # Current point
                colors.append('#FF0000')  # Bright red
                sizes.append(10)
                symbols.append('circle')
            elif j > 0 and gibbs_accepts[j - 1]:  # Accepted points
                colors.append('#FF9100')  # Bright orange for Gibbs (middle)
                sizes.append(8)
                symbols.append('circle')
            else:  # Rejected points
                colors.append('#FF1744')  # Bright red for rejected
                sizes.append(6)
                symbols.append('cross')

        # Add trail including current point
        frame_data.append(
            go.Scatter(x=gibbs_samples[trail_start:i + 1, 0],
                       y=gibbs_samples[trail_start:i + 1, 1],
                       mode='markers',
                       marker=dict(color=colors, size=sizes, symbol=symbols),
                       showlegend=False))

        # Add trail data for Metropolis-Hastings (leftmost)
        colors = []
        sizes = []
        symbols = []

        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i + 1):  # Include current point in trail
            if j == i:  # Current point
                colors.append('#FF0000')  # Bright red
                sizes.append(10)
                symbols.append('circle')
            elif j > 0 and mh_accepts[j - 1]:  # Accepted points
                colors.append('#00E676')  # Bright green for MH (leftmost)
                sizes.append(8)
                symbols.append('circle')
            else:  # Rejected points
                colors.append('#FF1744')  # Bright red for rejected
                sizes.append(6)
                symbols.append('cross')

        # Add trail including current point
        frame_data.append(
            go.Scatter(x=mh_samples[trail_start:i + 1, 0],
                       y=mh_samples[trail_start:i + 1, 1],
                       mode='markers',
                       marker=dict(color=colors, size=sizes, symbol=symbols),
                       showlegend=False))

        # Add trail data for Simulated Annealing (rightmost)
        colors = []
        sizes = []
        symbols = []

        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i + 1):  # Include current point in trail
            if j == i:  # Current point
                colors.append('#FF0000')  # Bright red
                sizes.append(10)
                symbols.append('circle')
            elif j > 0 and sa_accepts[j - 1]:  # Accepted points
                colors.append('#2979FF')  # Bright blue for SA (rightmost)
                sizes.append(8)
                symbols.append('circle')
            else:  # Rejected points
                colors.append('#FF1744')  # Bright red for rejected
                sizes.append(6)
                symbols.append('cross')

        # Add trail including current point
        frame_data.append(
            go.Scatter(x=sa_samples[trail_start:i + 1, 0],
                       y=sa_samples[trail_start:i + 1, 1],
                       mode='markers',
                       marker=dict(color=colors, size=sizes, symbol=symbols),
                       showlegend=False))

        # Create frame
        frames.append(
            go.Frame(data=frame_data,
                     name=str(i - 1),
                     traces=[1, 2, 3, 4, 5, 6]))

    # Add contours to all subplots
    for col in range(1, 4):
        fig.add_trace(go.Contour(z=z_values,
                                 x=x_grid[0, :],
                                 y=y_grid[:, 0],
                                 colorscale='Viridis',
                                 opacity=0.4,
                                 showscale=False,
                                 contours=dict(
                                     showlabels=False,
                                     coloring='fill',
                                 )),
                      row=1,
                      col=col)

    # Add initial points to the figure
    for col, (samples, color_name) in enumerate(
            zip(
                [gibbs_samples, sa_samples, mh_samples
                 ],  # Orange(Gibbs), Green(SA), Blue(MH)
                [
                    'rgba(255, 127, 14, 1)', 'rgba(44, 160, 44, 1)',
                    'rgba(31, 119, 180, 1)'
                ]),
            1):
        fig.add_trace(go.Scatter(x=[samples[0, 0]],
                                 y=[samples[0, 1]],
                                 mode='markers',
                                 marker=dict(color='red', size=10),
                                 showlegend=False),
                      row=1,
                      col=col)

    fig.frames = frames

    return frames


def add_annotations(fig):
    """
    Add informative annotations to the MCMC visualization figure.
    
    This function adds method-specific annotations to each subplot in the figure,
    providing insights about the different MCMC algorithms being visualized.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure object to which annotations will be added.
        The figure should have three subplots (one for each MCMC method).
        
    Returns
    -------
    plotly.graph_objects.Figure
        The modified figure with added annotations.
    """
    # Add annotations for each method
    annotations = [
        # Metropolis-Hastings
        dict(x=0.17,
             y=0.95,
             xref='paper',
             yref='paper',
             text='Random walk proposals<br>Isotropic jumps',
             showarrow=False,
             font=dict(size=10),
             bgcolor='rgba(255, 255, 255, 0.7)',
             bordercolor='rgba(0, 0, 0, 0.5)',
             borderwidth=1,
             borderpad=4),

        # Gibbs Sampling
        dict(x=0.5,
             y=0.95,
             xref='paper',
             yref='paper',
             text='Samples one dimension at a time<br>Always accepts moves',
             showarrow=False,
             font=dict(size=10),
             bgcolor='rgba(255, 255, 255, 0.7)',
             bordercolor='rgba(0, 0, 0, 0.5)',
             borderwidth=1,
             borderpad=4),

        # Simulated Annealing
        dict(x=0.83,
             y=0.95,
             xref='paper',
             yref='paper',
             text='Decreasing temperature<br>More exploration initially',
             showarrow=False,
             font=dict(size=10),
             bgcolor='rgba(255, 255, 255, 0.7)',
             bordercolor='rgba(0, 0, 0, 0.5)',
             borderwidth=1,
             borderpad=4)
    ]

    fig.update_layout(annotations=annotations)

    return fig
