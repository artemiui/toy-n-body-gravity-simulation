# toy-n-body-gravity-simulation
---

# Gravitational Model

The attraction between cosmological bodies is governed by a set of physical laws that are mostly relativistic, in other words, Einstein-based. Those are too complex, and require extremely powerful calculations to simulate. Although some tools are available now, I want to keep things simple for a first time. In this case, I develop a simple physical N-body simulation based on Newton’s equations on gravity, simulating the cumulative orbit of any *n* amount of bodies in a given plane.

---

## Gravitational Model

The gravitational force between two particles *i* and *j* is given by:

<img width="156" height="59" alt="image" src="https://github.com/user-attachments/assets/9e187d11-466b-4e09-8f06-f57a5f67a069" />

To prevent the formation of singularities, an outcome of gravitational collapse of a system of bodies, we introduce *ε*. This acts as a softening variable. This softening, however, only smooths out short-distance interactions. When I add more bodies, say *n = 500*, particles tend to collapse into the center of mass, and then fling away. This is because in physical cosmological models, dark matter halo is introduced in its models to prevent this exact problem from happening. Without this factor, particles can gain enough kinetic energy during interactions to drift or escape indefinitely.

We therefore introduce linear restoring force is sometimes to mimic the confining effect of a dark matter halo. This force pulls each particle toward the origin, similar to a spring:

<img width="148" height="42" alt="image" src="https://github.com/user-attachments/assets/93a7c166-de78-4437-bf80-f5d31574a3c1" />


Where *k* is a constant representing the strength of the dark matter potential (units: kpc/Myr²), and *r* is the position vector of the particle relative to the origin. This force we mimic is correspondent to harmonic potential energy:

<img width="167" height="50" alt="image" src="https://github.com/user-attachments/assets/dc745e67-8c11-425d-bfd1-c3d177a7df39" />


Where *r = ||r||* is the distance from the center. This potential ensures that particles are bound to a central region and prevents dispersion in low-particle-number simulations.

Thus, the total force acting on a particle *i* in the simulation is given by the sum of pairwise softened gravitational forces and a linear restoring force from a dark matter potential.
