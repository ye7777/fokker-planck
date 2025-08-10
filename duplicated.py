import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal

def run_duplicated_simulation():
    """
    Sets up and runs the Fokker-Planck simulation for the duplicated Langevin system.
    """
    # --- 1. Parameter Setup ---
    # Spatial Grid
    N = 101  # Number of grid points for both x1 and x2
    x_min, x_max = -4.0, 4.0
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Time Grid
    T = 8.0  # Total simulation time
    dt = 0.001 # Time step size
    Nt = int(T / dt)

    # Physical Parameters
    # For f(x) = -x, the force constant k=1
    k = 1.0
    # Diffusion coefficient D
    D = 0.5
    # Coupling constant gamma (breaks detailed balance)
    gamma = 1.0 # Set to 0 to recover two independent processes

    # --- 2. Initial Condition ---
    # Start with a narrow 2D Gaussian distribution centered off-origin
    # to better visualize the drift and rotation.
    mu_0 = [1.5, 1.5]
    cov_0 = [[0.05, 0], [0, 0.05]]
    P = multivariate_normal.pdf(np.stack((X, Y), axis=-1), mean=mu_0, cov=cov_0)
    P /= np.sum(P) * dx * dy  # Normalize

    # --- 3. Plotting and Animation Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Duplicated Langevin FPE (γ = {gamma})")
    ax.set_xlabel("Position (x1)")
    ax.set_ylabel("Position (x2)")

    # Use imshow for 2D heatmap visualization
    im = ax.imshow(P.T, cmap='viridis', origin='lower',
                   extent=[x_min, x_max, x_min, x_max],
                   vmin=0, vmax=np.max(P)*1.1) # Transpose P for correct axis orientation
    fig.colorbar(im, ax=ax, label="Probability Density P(x1, x2, t)")
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

    # --- 4. Update Function for Animation ---
    def update(frame):
        """
        Calculates the state of the system at the next time step using Finite Differences.
        """
        nonlocal P
        P_current = P.copy()

        # Loop over interior grid points
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # --- Fokker-Planck Equation Terms ---
                # f(x) = -k*x
                f_x1 = -k * X[i, j]
                f_x2 = -k * Y[i, j]

                # Drift terms: -∂/∂x_i [A_i * P]
                # Using central differences for ∂(stuff)/∂x
                drift_x1 = ( (f_x1 + gamma * f_x2) * P_current[i,j] ) * 0 # This is a placeholder, we need the derivative of the product
                drift_x1_term_full = ((-k*X[i+1,j] + gamma*(-k*Y[i+1,j])) * P_current[i+1,j] - \
                                      (-k*X[i-1,j] + gamma*(-k*Y[i-1,j])) * P_current[i-1,j]) / (2*dx)

                drift_x2 = ( (f_x2 - gamma * f_x1) * P_current[i,j] ) * 0 # This is a placeholder
                drift_x2_term_full = ((-k*Y[i,j+1] - gamma*(-k*X[i,j+1])) * P_current[i,j+1] - \
                                      (-k*Y[i,j-1] - gamma*(-k*X[i,j-1])) * P_current[i,j-1]) / (2*dy)

                # Diffusion terms: D * ∂²P/∂x_i²
                diffusion_x1 = D * (P_current[i+1, j] - 2*P_current[i, j] + P_current[i-1, j]) / (dx**2)
                diffusion_x2 = D * (P_current[i, j+1] - 2*P_current[i, j] + P_current[i, j-1]) / (dy**2)

                # Combine terms to update P
                P[i, j] = P_current[i, j] + dt * ( -drift_x1_term_full - drift_x2_term_full + diffusion_x1 + diffusion_x2 )

        # Enforce boundary conditions (zero probability at the edges)
        P[0, :] = P[-1, :] = P[:, 0] = P[:, -1] = 0

        # Re-normalize to conserve total probability
        P /= np.sum(P) * dx * dy

        # Update the plot data
        im.set_data(P.T)
        im.set_clim(vmin=0, vmax=np.max(P)) # Adjust color limits for visibility
        time_text.set_text(f'Time = {frame * dt:.2f} s')
        return im, time_text

    # --- 5. Run Animation ---
    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=1, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    try:
        import numpy
        import matplotlib
        import scipy
    except ImportError as e:
        print(f"Error: A required library is not installed. {e}")
        print("Please install necessary libraries using: pip install numpy matplotlib scipy")
    else:
        run_duplicated_simulation()
