import cupy as cp
from scipy import c, G
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import seaborn as sns

# config 
N_PARTICLES = 100       # no. of stars
SIM_STEPS = 10000         # simulation steps
PLOT_EVERY = 200
DT = 0.01               # time step [Myr]
SOFTENING = 1.0         # gravitational softening [kpc]
DARK_MATTER_FORCE = 0.002  # [kpc/Myr^2]
AXIS_SCALE = 10 + N_PARTICLES / 100  # scales with particle count

# some constants
KPC_TO_KM = 3.086e16    # 1 kpc to km
MYR_TO_S = 3.154e13     # 1 Myr to seconds
SOLAR_MASS_TO_KG = 1.989e30  # solar mass to kg

PARTICLE_CMAP = sns.color_palette("PuBu", as_cmap=True)

# physics core
def compute_forces(positions, masses, softening):
    N = positions.shape[0]
    forces = cp.zeros_like(positions, dtype=cp.float32)
    
    dx = positions[:, None, 0] - positions[None, :, 0]
    dy = positions[:, None, 1] - positions[None, :, 1]
    dz = positions[:, None, 2] - positions[None, :, 2]
    
    r_squared = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3 = r_squared ** (-1.5)
    inv_r3 *= G * masses[None, :] / (KPC_TO_KM**3/MYR_TO_S**2)  # Unit conversion
    
    forces[:, 0] = -cp.sum(dx * inv_r3, axis=1)
    forces[:, 1] = -cp.sum(dy * inv_r3, axis=1)
    forces[:, 2] = -cp.sum(dz * inv_r3, axis=1)
    
    forces -= DARK_MATTER_FORCE * positions
    return forces

def simulation_step(positions, velocities, masses, dt):
    forces = compute_forces(positions, masses, SOFTENING)
    velocities += forces * dt
    positions += velocities * dt
    return positions, velocities, masses

# mass-color mapping
def masses_to_colors(masses):
    """Convert masses to colors using Seaborn palette"""
    norm_masses = (masses - np.min(masses)) / (np.max(masses) - np.min(masses) + 1e-10)
    return PARTICLE_CMAP(norm_masses)

# sim
def run_simulation():
    # calculations initialize on GPU
    cp.random.seed(42)
    positions = cp.random.randn(N_PARTICLES, 3).astype(cp.float32) * (AXIS_SCALE/3)
    velocities = cp.random.randn(N_PARTICLES, 3).astype(cp.float32) * 0.1
    masses = (cp.random.rand(N_PARTICLES).astype(cp.float32) * 10 + 1)
    
    history = []
    mass_history = []
    color_history = []
    strain_history = []
    energy_history = []
    
    for step in tqdm(range(SIM_STEPS), desc="Simulating"):
        positions, velocities, masses = simulation_step(positions, velocities, masses, DT)
        
        if step % PLOT_EVERY == 0:
            # convert to cpu for visualization
            pos_cpu = cp.asnumpy(positions)
            mass_cpu = cp.asnumpy(masses)
            vel_cpu = cp.asnumpy(velocities)
            colors = masses_to_colors(mass_cpu)
            
            # gravitational wave strain calculation
            if len(mass_cpu) >= 2:
                idx = np.argsort(mass_cpu)[-2:]
                m1, m2 = mass_cpu[idx[0]], mass_cpu[idx[1]]
                p1, p2 = pos_cpu[idx[0]], pos_cpu[idx[1]]
                v1, v2 = vel_cpu[idx[0]], vel_cpu[idx[1]]
                
                r = np.linalg.norm(p2 - p1) * KPC_TO_KM
                v = np.linalg.norm(v2 - v1) * (KPC_TO_KM/MYR_TO_S)
                mu = (m1 * m2 / (m1 + m2)) * SOLAR_MASS_TO_KG
                strain = (4 * G * mu * v**2) / (c**4 * r)
                strain_history.append(strain)
            else:
                strain_history.append(0)
            
            # energy calculation
            kinetic = 0.5 * cp.sum(masses * cp.sum(velocities**2, axis=1))
            kinetic = float(cp.asnumpy(kinetic))
            
            potential = 0
            chunk_size = 512
            for i in range(0, N_PARTICLES, chunk_size):
                chunk_pos = positions[i:i+chunk_size]
                chunk_mass = masses[i:i+chunk_size]
                diff = chunk_pos[:, None, :] - positions[None, :, :]
                r = cp.sqrt(cp.sum(diff**2, axis=2) + SOFTENING)
                potential -= 0.5 * cp.sum(chunk_mass[:, None] * masses[None, :] / r)
            
            potential = float(cp.asnumpy(potential))
            total_energy = kinetic + potential
            energy_history.append(total_energy)
            
            history.append(pos_cpu)
            mass_history.append(mass_cpu)
            color_history.append(colors)
            
            # clear gpu memory periodically for optimization
            if step % 1000 == 0:
                cp.get_default_memory_pool().free_all_blocks()
    
    return history, mass_history, color_history, np.array(strain_history), np.array(energy_history)

# vis
def create_interactive_visualization(history, mass_history, color_history, strain_history, energy_history):
    plt.style.use('dark_background')
    sns.set_style("dark")
    
    fig = plt.figure(figsize=(16, 10), facecolor='black') 
    
    gs = fig.add_gridspec(
        3, 2, 
        height_ratios=[2, 1, 0.2],  
        width_ratios=[1.5, 1],     
        left=0.08, right=0.95,      
        bottom=0.12, top=0.95,      
        hspace=0.3, wspace=0.25     
    )
    
    ax_3d = fig.add_subplot(gs[0:2, 0], projection='3d')  
    ax_strain = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    
    ax_3d.set_facecolor('black')
    ax_3d.grid(False)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    
    ax_3d.xaxis.label.set_color('white')
    ax_3d.yaxis.label.set_color('white')
    ax_3d.zaxis.label.set_color('white')
    ax_3d.tick_params(axis='x', colors='white', labelsize=9)  
    ax_3d.tick_params(axis='y', colors='white', labelsize=9)
    ax_3d.tick_params(axis='z', colors='white', labelsize=9)
    
    ax_3d.set_xlim(-AXIS_SCALE, AXIS_SCALE)
    ax_3d.set_ylim(-AXIS_SCALE, AXIS_SCALE)
    ax_3d.set_zlim(-AXIS_SCALE, AXIS_SCALE)
    
    ax_3d.set_xlabel('X [kpc]', labelpad=10, fontsize=10)
    ax_3d.set_ylabel('Y [kpc]', labelpad=10, fontsize=10)
    ax_3d.set_zlabel('Z [kpc]', labelpad=10, fontsize=10)

    ax_strain.set_facecolor('#111111')
    ax_strain.grid(False)
    ax_strain.tick_params(colors='white', labelsize=8)  
    for spine in ax_strain.spines.values():
        spine.set_edgecolor('white')
    
    ax_strain.set_title('Gravitational Wave Strain', color='white', pad=6, fontsize=10)
    ax_strain.set_ylabel(f'Strain (×10$^{-21}$)', color='white', fontsize=10, labelpad=12)
    ax_strain.set_yscale('log')
    ax_strain.set_xlabel('Time [Myr]', color='white', fontsize=10, labelpad=12)

    for text in ax_strain.get_yticklabels() + ax_strain.get_xticklabels():
        text.set_color('white')
    
    ax_energy.set_facecolor('#111111')
    ax_energy.grid(False)
    ax_energy.tick_params(colors='white', labelsize=8) 
    for spine in ax_energy.spines.values():
        spine.set_edgecolor('white')
    
    ax_energy.set_title('Relative Energy Conservation', color='white', pad=6, fontsize=10)
    ax_energy.set_xlabel('Time [Myr]', color='white', fontsize=9)
    ax_energy.set_ylabel('Energy / Initial', color='white', fontsize=9)  
    
    # convert time steps to myr
    time_myr = np.arange(len(strain_history)) * DT * PLOT_EVERY
    
    # initial data with optimized marker size
    frame0 = history[0]
    scat = ax_3d.scatter(
        frame0[:,0], frame0[:,1], frame0[:,2],
        s=3,  # marker size
        c=color_history[0],
        alpha=1.0,
        edgecolors='none',
        depthshade=False
    )
    
    strain_line = ax_strain.plot(time_myr, strain_history,
                            color='cyan', lw=1.5,
                            label=f'Max: {np.max(strain_history):.1e}')[0]
    strain_legend = ax_strain.legend(loc='upper right',
                                    facecolor='none',  
                                    edgecolor='none',  
                                    labelcolor='white',
                                    fontsize=8)
    
    # plot relative energy
    relative_energy = energy_history / energy_history[0]
    energy_line = ax_energy.plot(time_myr, relative_energy,
                            color='yellow', lw=1.5,
                            label=f'Final: {relative_energy[-1]:.4f}')[0]
    energy_legend = ax_energy.legend(loc='upper right', 
                                    facecolor='none',  
                                    edgecolor='none',  
                                    labelcolor='white',
                                    fontsize=8)
    ax_energy.axhline(1.0, color='red', linestyle='--', alpha=0.5)

    # compact info text
    initial_com = np.average(history[0], axis=0, weights=mass_history[0])
    final_com = np.average(history[-1], axis=0, weights=mass_history[-1])
    drift_distance = np.linalg.norm(final_com - initial_com)
        
    info_text = (
        f"CoM: ({final_com[0]:.1f}, {final_com[1]:.1f}, {final_com[2]:.1f}) kpc | "
        f"Drift: {drift_distance:.1f} kpc | "
        f"Max Strain: {np.max(strain_history):.1e} | "
        f"Energy: {relative_energy[-1]:.5f}"
    )
        
    ax_info.text(0.5, 0.5, info_text,
                color='white', ha='center', va='center',
                fontfamily='monospace', fontsize=10,
                bbox=dict(facecolor='#111111', edgecolor='white', alpha=0.7))
    
    # set consistent x-limits
    for ax in [ax_strain, ax_energy]:
        ax.set_xlim(0, time_myr[-1])
    
    # add current time indicators
    current_time_strain = ax_strain.axvline(0, color='white', alpha=0.5, lw=1)
    current_time_energy = ax_energy.axvline(0, color='white', alpha=0.5, lw=1)
    
    # compact controls
    ax_slider = plt.axes([0.25, 0.06, 0.5, 0.02], facecolor='#111111')  # Raised slider
    time_slider = Slider(
        ax=ax_slider,
        label='Time [Myr]',
        valmin=0,
        valmax=time_myr[-1],
        valinit=0,
        valstep=DT*PLOT_EVERY,
        color='#444444'
    )
    
    ax_play = plt.axes([0.45, 0.02, 0.1, 0.03])  # Smaller play button
    play_button = Button(ax_play, '▶ Play', color='#333333', hovercolor='#444444')
    
    def update(val):
        frame = int(val / (DT * PLOT_EVERY))
        if frame >= len(history):
            frame = len(history) - 1
        
        scat._offsets3d = (history[frame][:,0], history[frame][:,1], history[frame][:,2])
        scat.set_color(color_history[frame])
        
        current_time = time_myr[frame]
        current_time_strain.set_xdata([current_time, current_time])
        current_time_energy.set_xdata([current_time, current_time])
        
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    
    def play(event):
        for i in range(len(time_myr)):
            time_slider.set_val(time_myr[i])
            plt.pause(0.05)
    
    play_button.on_clicked(play)
    
    plt.show()
    
# ================= Main =================
if __name__ == "__main__":
    print(f"Running simulation with {N_PARTICLES} particles...")
    results = run_simulation()
    
    print("Launching visualization...")
    create_interactive_visualization(*results)