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

The model uses random sequential updating: N random single-site updates constitute one sweep. The transition rules are:
- **S → I** with probability `p_S`, if at least one nearest neighbour is infected.
- **I → R** with probability `p_I`, regardless of neighbours.
- **R → S** with probability `p_R`, regardless of neighbours.

### Arguments
- `n`, Lattice size (n x n), Default = 50
- `steps`, Number of animation steps, Default = 1000
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

**Average** — sweeps `p_S` and `p_R` (with `p_I = 0.5` fixed) and records the mean infected fraction, producing a 2D phase diagram heatmap.
```
python sirs.py --mode mea --measure average
```

**Variance** — sweeps `p_S` from 0.2 to 0.5 (with `p_I = p_R = 0.5` fixed) and records the normalised variance with bootstrap error bars, used to locate the phase transition.
```
python sirs.py --mode mea --measure variance
```

**Immunity** — sweeps the vaccinated fraction from 0 to 1 (with `p_S = p_I = p_R = 0.5` fixed) and records the mean infected fraction, used to find the herd immunity threshold.
```
python sirs.py --mode mea --measure immunity
```

---

## Outputs
All outputs are saved relative to the script's directory:
```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```
