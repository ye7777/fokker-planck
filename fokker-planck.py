# -*- coding: utf-8 -*-

"""
Fokker-Planck Equation Simulation for the Ornstein-Uhlenbeck Process.

This script simulates the time evolution of the probability density function p(x, t)
for a particle governed by the stochastic differential equation:
dx = -x*dt + dW

The corresponding Fokker-Planck equation is:
∂p/∂t = ∂/∂x[x*p] + (1/2) * ∂²p/∂x²
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

def run_simulation():
    """
    Sets up and runs the Fokker-Planck simulation and animation.
    """
    # --- 1. Parameter Setup ---
    Nx = 201  # Number of spatial grid points
    x_min, x_max = -5.0, 5.0  # Spatial domain
    dx = (x_max - x_min) / (Nx - 1)  # Spatial step size
    x = np.linspace(x_min, x_max, Nx)  # Spatial grid

    T = 5.0  # Total simulation time
    dt = 0.001 # Time step size
    Nt = int(T / dt)  # Number of time steps

    # Diffusion coefficient D (from the term (1/2)*d²p/dx²)
    D = 0.5

    # --- 2. Initial Condition ---
    # Start with a narrow Gaussian distribution centered at x=0.
    # This approximates a particle starting at a specific point.
    mu_0 = 0.0
    sigma_0 = 0.2
    p = norm.pdf(x, mu_0, sigma_0)
    p /= np.sum(p) * dx  # Normalize the initial probability density

    # --- 3. Plotting and Animation Setup ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Fokker-Planck Simulation: Ornstein-Uhlenbeck Process")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Probability Density p(x, t)")
    ax.grid(True)

    # Plot the simulation line (this will be updated)
    line, = ax.plot(x, p, 'b-', lw=2, label='p(x, t) (Simulation)')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Plot the theoretical stationary solution for comparison
    # For this process, it's a Gaussian with mean=0 and variance=D
    p_stationary = norm.pdf(x, 0, np.sqrt(D))
    ax.plot(x, p_stationary, 'r--', lw=2, label=f'Stationary Solution (N(0, {D}))')
    
    ax.legend(loc='upper right')
    ax.set_ylim(0, np.max(p_stationary) * 1.5)
    ax.set_xlim(x_min, x_max)

    # --- 4. Update Function for Animation ---
    def update(frame):
        """
        Calculates the state of the system at the next time step.
        This function is called for each frame of the animation.
        """
        nonlocal p  # Modify the 'p' from the outer scope
        
        # Create a copy to use for calculations based on the current step
        p_current = p.copy()
        
        # Apply the finite difference method across the spatial grid
        for i in range(1, Nx - 1):
            # First derivative term (drift): ∂(x*p)/∂x
            # Using the central difference scheme
            drift_term = (x[i+1] * p_current[i+1] - x[i-1] * p_current[i-1]) / (2 * dx)
            
            # Second derivative term (diffusion): ∂²p/∂x²
            # Using the central difference scheme
            diffusion_term = (p_current[i+1] - 2 * p_current[i] + p_current[i-1]) / (dx**2)
            
            # Update the probability density at point i
            p[i] = p_current[i] + dt * (drift_term + D * diffusion_term)
            
        # Enforce boundary conditions (probability is zero at the edges)
        p[0] = p[Nx-1] = 0
        
        # Re-normalize to conserve total probability (accounts for numerical errors)
        p /= np.sum(p) * dx
        
        # Update the plot data
        line.set_ydata(p)
        time_text.set_text(f'Time = {frame * dt:.2f} s')
        return line, time_text

    # --- 5. Run Animation ---
    # Create the animation object
    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=1, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    # Ensure necessary libraries are installed
    try:
        import numpy
        import matplotlib
        import scipy
    except ImportError as e:
        print(f"Error: A required library is not installed. {e}")
        print("Please install necessary libraries using: pip install numpy matplotlib scipy")
    else:
        run_simulation()