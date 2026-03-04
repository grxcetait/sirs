# SIRS Model — `sirs.py`

Python script for simulating the Susceptible-Infected-Recovered-Susceptible (SIRS) epidemic model on a 2D square lattice with periodic boundary conditions.

## sirs.py

This script has all the functions and classes to either run an animation or measurements of the simulation.
The user needs to put in different arguments and customise the animation or measurement conditions.

The lattice is initialised randomly with states drawn from normalised transition probabilities. Each cell can be in one of four states:
- `-2` Vaccinated (permanently immune)
- `-1` Infected
- `0` Susceptible
- `1` Recovered

The model uses random sequential updating where the transition rules are:
- Susceptible to infected with probability `p_S`.
- Infected to recovered with probability `p_I`.
- Recovered to susceptible with probability `p_R`.

### Arguments

- `n`, Lattice size (n x n), Default = 50
- `steps`, Number of simulation steps, Default = 1000
- `p_S`, Probability of S → I (infection), Default = 0.3
- `p_I`, Probability of I → R (recovery), Default = 0.5
- `p_R`, Probability of R → S (immunity loss), Default = 0.2
- `mode`, Mode of `ani` (animation) or `mea` (measurements), Default = `ani`
- `measure`, Measurement type: `average`, `variance`, or `immunity`, Default = `average`

### Output

All outputs are saved relative to the script's directory:

```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```

## Command line examples

### Animation (`--mode ani`)

Display an animation of the SIRS model using the provided probabilities.

```
python sirs.py --mode ani --p_S 0.5 --p_I 0.5 --p_R 0.5
python sirs.py --mode ani --p_S 0.0 --p_I 0.5 --p_R 0.5
python sirs.py --mode ani --p_S 0.3 --p_I 0.5 --p_R 0.05
```

### Measurements (`--mode mea`)

Run data collection depending on the `--measure` choice:

**Average** — iterates through `p_S` and `p_R` (with `p_I = 0.5` fixed) and records the mean infected fraction, producing a 2D phase diagram heatmap.

```
python sirs.py --mode mea --measure average
```

**Variance** — iterates through `p_S` from 0.2 to 0.5 (with `p_I = p_R = 0.5` fixed) and records the normalised variance with bootstrap error bars.

```
python sirs.py --mode mea --measure variance
```

**Immunity** — iterates the vaccinated fraction from 0 to 1 (with `p_S = p_I = p_R = 0.5` fixed) and records the mean infected fraction.

```
python sirs.py --mode mea --measure immunity
```


