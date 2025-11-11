import numpy as np
from numba import njit # gpu acceleration
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import seaborn as sns
from tqdm import tqdm

# config
N_PARTICLES = 100       # no. of bodies
SIM_STEPS = 200         # simulation steps
DT = 0.01               # time step
THETA = 0.7             # barnes-Hut opening angle
SOFTENING = 0.1         # gravitational softening
PLOT_EVERY = 5          # store positions every N steps

# physics core
@njit
def compute_forces(positions, masses, theta, G, softening):
    forces = np.zeros_like(positions)
    for i in range(len(positions)):
        force = np.zeros(2)
        for j in range(len(positions)):
            if i == j: continue
            r = positions[j] - positions[i]
            r_dist = np.sqrt(r[0]**2 + r[1]**2 + softening**2)
            force += G * masses[j] * r / (r_dist**3)
        forces[i] = force
    return forces

@njit
def simulation_step(positions, velocities, masses, dt, theta=0.5, G=1.0):
    forces = compute_forces(positions, masses, theta, G, SOFTENING)
    new_velocities = velocities + forces * dt
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities

# simulation
def run_simulation():
    np.random.seed(42)
    positions = np.random.randn(N_PARTICLES, 2) * 10
    velocities = np.random.randn(N_PARTICLES, 2) * 0.1
    masses = np.random.rand(N_PARTICLES) * 10 + 1
    
    history = np.zeros((SIM_STEPS//PLOT_EVERY, N_PARTICLES, 2))
    
    for step in tqdm(range(SIM_STEPS), desc="Simulating"):
        positions, velocities = simulation_step(positions, velocities, masses, DT, THETA)
        if step % PLOT_EVERY == 0:
            history[step//PLOT_EVERY] = positions
    
    return history, masses

# vis
def interactive_visualization(history, masses):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(bottom=0.2)
    
    size_norm = masses / masses.max() * 100  # here i normalize masses for coloring
    cmap = plt.cm.plasma
    norm = plt.Normalize(masses.min(), masses.max())
    colors = cmap(norm(masses))
    
    scat = ax.scatter(history[0,:,0], history[0,:,1], 
                     s=size_norm, c=colors, alpha=0.8)
    
    line_collections = []
    for i in range(0, N_PARTICLES, max(1, N_PARTICLES//50)):
        x = history[:, i, 0]
        y = history[:, i, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', 
                          norm=plt.Normalize(0, len(history)),
                          alpha=0.2, linewidth=0.7, zorder=1)
        lc.set_array(np.arange(len(history)))
        ax.add_collection(lc)
        line_collections.append(lc)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Particle Mass', rotation=270, labelpad=20)
    
# some slider and button controls for config
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    time_slider = Slider(ax_slider, 'Time Step', 0, len(history)-1, valinit=0, valstep=1)
    
    def update(frame):
        frame = int(frame)
        scat.set_offsets(history[frame])
        
        for lc in line_collections:
            lc.set_alpha(np.linspace(0.1, 0.3, len(history))[frame])
        
        fig.canvas.draw_idle()  
        return scat,
        if frame < len(history)-1:
            dx = history[frame+1,:,0] - history[frame,:,0]
            dy = history[frame+1,:,1] - history[frame,:,1]
            ax.quiver(history[frame,:,0], history[frame,:,1], 
                    dx, dy, color='white', scale=50, width=0.)
        
    time_slider.on_changed(update)
    
    ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
    play_button = Button(ax_button, '▶ Play', color='lightgray')
    
    def play_animation(event):
        for i in range(len(history)):
            time_slider.set_val(i)
            plt.pause(0.01)
    
    play_button.on_clicked(play_animation)
    
    ax.set_xlim(history[:,:,0].min()-1, history[:,:,0].max()+1)
    ax.set_ylim(history[:,:,1].min()-1, history[:,:,1].max()+1)
    ax.set_title(f"N-Body Simulation (N={N_PARTICLES})", pad=20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_facecolor('black')
    
    plt.show()

# sim
if __name__ == "__main__":
    print("Running simulation...")
    history, masses = run_simulation()
    
    print("Launching interactive visualization...")
    interactive_visualization(history, masses)