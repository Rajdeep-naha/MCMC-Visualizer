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
                           mh_accepts, gibbs_accepts, sa_accepts, 
                           mh_proposed, gibbs_proposed, sa_proposed,
                           trail_length):
    """
    Create animation frames for the MCMC visualization.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add frames to
    mh_samples : array-like
        Samples from Metropolis-Hastings
    gibbs_samples : array-like
        Samples from Gibbs Sampling
    sa_samples : array-like
        Samples from Simulated Annealing
    mh_accepts : array-like
        Boolean array indicating whether each Metropolis-Hastings proposal was accepted
    gibbs_accepts : array-like
        Boolean array indicating whether each Gibbs Sampling proposal was accepted
    sa_accepts : array-like
        Boolean array indicating whether each Simulated Annealing proposal was accepted
    trail_length : int
        Length of the trail to show
    
    Returns:
    --------
    list
        List of frames for animation
    """
    frames = []
    n_iterations = len(mh_accepts)
    
    # Create frames for animation
    for i in range(1, n_iterations + 1):
        frame_data = []
        
        # Calculate trail start point
        trail_start = max(0, i - trail_length)
        
        # Make copies of the samples to avoid references
        mh_frame_samples = mh_samples.copy()
        gibbs_frame_samples = gibbs_samples.copy()
        sa_frame_samples = sa_samples.copy()
        
        # Add trail data for Metropolis-Hastings
        colors = []
        sizes = []
        symbols = []
        
        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i):
            if mh_accepts[j - 1]:
                # Accepted points
                colors.append('rgba(31, 119, 180, 0.9)')
                sizes.append(8)
                symbols.append('circle')
            else:
                # Rejected points
                colors.append('rgba(255, 0, 0, 0.7)')
                sizes.append(6)
                symbols.append('cross')
        
        # Add trail - using mh_frame_samples to avoid reference issues
        frame_data.append(
            go.Scatter(
                x=mh_frame_samples[trail_start:i, 0],
                y=mh_frame_samples[trail_start:i, 1],
                mode='markers',
                marker=dict(color=colors, size=sizes, symbol=symbols),
                showlegend=False
            )
        )
        
        # Add current point (red dot at current position) - using mh_frame_samples to avoid reference issues
        frame_data.append(
            go.Scatter(
                x=[mh_frame_samples[i, 0]],
                y=[mh_frame_samples[i, 1]],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False
            )
        )
        
        # Add proposed point if it was rejected (showing where it tried to go)
        if i > 0 and not mh_accepts[i-1]:
            frame_data.append(
                go.Scatter(
                    x=[mh_proposed[i-1, 0]],
                    y=[mh_proposed[i-1, 1]],
                    mode='markers',
                    marker=dict(color='rgba(255, 0, 0, 0.5)', size=8, symbol='cross'),
                    showlegend=False
                )
            )
        
        # Add trail data for Gibbs Sampling
        colors = []
        sizes = []
        symbols = []
        
        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i):
            if gibbs_accepts[j - 1]:
                # Accepted points
                colors.append('rgba(255, 127, 14, 0.9)')
                sizes.append(8)
                symbols.append('circle')
            else:
                # Rejected points
                colors.append('rgba(255, 0, 0, 0.7)')
                sizes.append(6)
                symbols.append('cross')
        
        # Add trail (using gibbs_frame_samples to avoid reference issues)
        frame_data.append(
            go.Scatter(
                x=gibbs_frame_samples[trail_start:i, 0],
                y=gibbs_frame_samples[trail_start:i, 1],
                mode='markers',
                marker=dict(color=colors, size=sizes, symbol=symbols),
                showlegend=False
            )
        )
        
        # Add current point (red dot at current position) - using gibbs_frame_samples to avoid reference issues
        frame_data.append(
            go.Scatter(
                x=[gibbs_frame_samples[i, 0]],
                y=[gibbs_frame_samples[i, 1]],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False
            )
        )
        
        # Add proposed point if it was rejected (showing where it tried to go)
        if i > 0 and not gibbs_accepts[i-1]:
            frame_data.append(
                go.Scatter(
                    x=[gibbs_proposed[i-1, 0]],
                    y=[gibbs_proposed[i-1, 1]],
                    mode='markers',
                    marker=dict(color='rgba(255, 0, 0, 0.5)', size=8, symbol='cross'),
                    showlegend=False
                )
            )
        
        # Add trail data for Simulated Annealing
        colors = []
        sizes = []
        symbols = []
        
        # Set different styles for accepted vs rejected points
        for j in range(trail_start, i):
            if sa_accepts[j - 1]:
                # Accepted points
                colors.append('rgba(44, 160, 44, 0.9)')
                sizes.append(8)
                symbols.append('circle')
            else:
                # Rejected points
                colors.append('rgba(255, 0, 0, 0.7)')
                sizes.append(6)
                symbols.append('cross')
        
        # Add trail (using sa_frame_samples to avoid reference issues)
        frame_data.append(
            go.Scatter(
                x=sa_frame_samples[trail_start:i, 0],
                y=sa_frame_samples[trail_start:i, 1],
                mode='markers',
                marker=dict(color=colors, size=sizes, symbol=symbols),
                showlegend=False
            )
        )
        
        # Add current point (red dot at current position) - using sa_frame_samples to avoid reference issues
        frame_data.append(
            go.Scatter(
                x=[sa_frame_samples[i, 0]],
                y=[sa_frame_samples[i, 1]],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False
            )
        )
        
        # Add proposed point if it was rejected (showing where it tried to go)
        if i > 0 and not sa_accepts[i-1]:
            frame_data.append(
                go.Scatter(
                    x=[sa_proposed[i-1, 0]],
                    y=[sa_proposed[i-1, 1]],
                    mode='markers',
                    marker=dict(color='rgba(255, 0, 0, 0.5)', size=8, symbol='cross'),
                    showlegend=False
                )
            )
        
        # Create frame - traces need to match the number of elements in frame_data
        # Since we've added additional traces for rejected proposals, need to update the traces indices
        trace_indices = list(range(len(frame_data)))
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(i - 1),
                traces=trace_indices
            )
        )
    
    # Add initial points to the figure
    for col, (samples, color_name) in enumerate(zip(
        [mh_samples, gibbs_samples, sa_samples],
        ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)']
    ), 1):
        fig.add_trace(
            go.Scatter(
                x=[samples[0, 0]],
                y=[samples[0, 1]],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False
            ),
            row=1, col=col
        )
    
    # Add frames to figure
    fig.frames = frames
    
    return frames

def add_annotations(fig):
    """
    Add annotations to the figure.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add annotations to
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with annotations
    """
    # Add annotations for each method
    annotations = [
        # Metropolis-Hastings
        dict(
            x=0.17,
            y=0.95,
            xref='paper',
            yref='paper',
            text='Random walk proposals<br>Isotropic jumps',
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            borderpad=4
        ),
        
        # Gibbs Sampling
        dict(
            x=0.5,
            y=0.95,
            xref='paper',
            yref='paper',
            text='Samples one dimension at a time<br>Always accepts moves',
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            borderpad=4
        ),
        
        # Simulated Annealing
        dict(
            x=0.83,
            y=0.95,
            xref='paper',
            yref='paper',
            text='Decreasing temperature<br>More exploration initially',
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            borderpad=4
        )
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig
