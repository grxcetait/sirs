#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 11:54:13 2026

@author: gracetait
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import argparse
import matplotlib.patches as mpatches
from numba import njit

@njit
def sirs_step_numba(lattice, i, j, n, p_S, p_I, p_R):
    """
    Determine the next state of a specific cell (i, j) using Monte Carlo rules.
    States: -2 (Immune), -1 (Infected), 0 (Susceptible), 1 (Recovered)
    
    Parameters
    ----------
    lattice : numpy.ndarray
        A 2D array of shape (n, n).
    i : int
        Lattice coordinate of the cell.
    j : int
        Lattice coordinate of the cell
    n : int
        Dimension of the square lattice (n x n)
    p_S : float
        Probability of infection from suscetible.
    p_I : float
        Probability of recovered from infected.
    p_R : float
        Probability of susceptible from recovered.

    Returns
    -------
    int
        The new state of the cell (-2, -1, 0, or 1).
    """
    
    # Obtain the site
    cell = lattice[i, j]
    
    # First, check if the cell is vaccinated
    if cell == -2: 
        
        # The cell remains vaccinated 
        return -2
        
    # Check if the cell is susceptible ...
    if cell == 0:
        
        # Count infected neighbors using periodic boundaries
        in_up = lattice[(i - 1) % n, j] == -1
        in_down = lattice[(i + 1) % n, j] == -1
        in_left = lattice[i, (j - 1) % n] == -1
        in_right = lattice[i, (j + 1) % n] == -1
        
        if in_up or in_down or in_left or in_right:
            
            # The cell will be infected with probability p_S
            if np.random.random() < p_S:
                return -1
            
        # Otherwise, the cell remains susceptible
        return 0
        
    # If the cell is infected ...
    elif cell == -1:
        
        # The cell will be recovered with probability p_I
        if np.random.random() < p_I:
            return 1
        
        # Otherwise, the cell remains infected
        return -1
        
    # If the cell is recovered ...
    else: 
        
        # The cell will be susceptible with probability p_S
        if np.random.random() < p_R:
            return 0
        
        # Otherwise, the cell remains recovered
        return 1

@njit
def sirs_sweep_numba(lattice, n, N, p_S, p_I, p_R):
    """
    Performs one full Monte Carlo Sweep (N random updates) across the lattice.

    Parameters
    ----------
    lattice : numpy.ndarray
        A 2D array of shape (n, n).
    n : int
        Dimension of the square lattice (n x n).
    n : int
        One sweep of the lattice (n x n).
    p_S : float
        Probability of infection from suscetible.
    p_I : float
        Probability of recovered from infected.
    p_R : float
        Probability of susceptible from recovered.

    Returns
    -------
    numpy.ndarray
        The updated lattice 
    """
    
    # Iterate N times through the lattice
    for _ in range(N):
        
        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        
        # Check for updates
        lattice[i, j] = sirs_step_numba(lattice, i, j, n, p_S, p_I, p_R)
        
    return lattice

class SIRS(object):
    """
    A class to represent a Susceptible-Infected-Recovered-Susceptible (SIRS)
    model on a 2D lattice.
    """
    
    def __init__(self, n, p_S, p_I, p_R):
        """
        Initialise the SIRS lattice parameters

        Parameters
        ----------
        n : int
            Dimension of the square lattice (n x n).
        p_S : float
            Probability of a Susceptible cell becoming Infected (S -> I).
        p_I : float
            Probability of an Infected cell becoming Recovered (I -> R).
        p_R : float
            Probability of a Recovered cell becoming Susceptible (R -> S).

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the lattice
        self.lattice = None
        self.p_S = p_S # Probability of susceptible to infected
        self.p_I = p_I # Probability of infected to recovered
        self.p_R = p_R # Probability of recovered to susceptible
        
    def initialise(self):
        """
        Randomly initialise the lattice states based on normalised transition probabilities.
        States are mapped as: -1 (Infected), 0 (Susceptible), 1 (Recovered).

        Returns
        -------
        None.

        """
        
        # Calculate the sum of probabilities
        p_tot = self.p_S + self.p_I + self.p_R
        
        # Calculate the normalised probabilities so they equal 1
        p_norm = [self.p_I / p_tot, self.p_S / p_tot, self.p_R / p_tot]
        
        # Create a two-dimensional square lattice according to probabilities
        # Where -1 is infected, 0 is susceptible, 1 is alive
        self.lattice = np.random.choice([-1, 0, 1], size = (self.n, self.n),
                                        p = p_norm).astype(np.int32)
            
    def update_lattice(self):
            """
            Perform one full Monte Carlo Sweep using Numba.
            """
            
            # Call Numba function
            self.lattice = sirs_sweep_numba(self.lattice, self.n, self.N,
                                            self.p_S, self.p_I, self.p_R)
    
    def count_infected(self):
        """
        Count the total number of infected cells currently in the lattice.

        Returns
        -------
        int
            Number of cells with state -1.

        """
    
        # Count and return the total number of infected cells in the lattice
        return np.count_nonzero(self.lattice == -1)
    
    def vaccinate(self, frac_immunity):
        """
        Randomly set a fraction of the lattice to a permanent immune state (-2).

        Parameters
        ----------
        frac_immunity : float
            The fraction of the total population to be vaccinated (0 to 1).

        Returns
        -------
        np.ndarray
            The updated lattice.

        """
        
        # Calculate the number of sites to vaccinate
        n_vaccinate = int(frac_immunity * self.N)
        
        # Pick unique inficies from a flattened version of the N sites
        # replace = False so the same site is not picked twice
        indices = np.random.choice(self.N, size = n_vaccinate, replace = False)
        
        # Iterate through the indicies
        for idx in indices:
            
            # Update the lattice
            i, j = divmod(idx, self.n)
            self.lattice[i, j] = -2
            

class Simulation(object):
    """
    A class to handle the execution, measurement, and visualisation 
    of the SIRS simulation.
    """
    
    def __init__(self, n, steps, p_S, p_I, p_R):
        """
        Initialise the simulation environment.

        Parameters
        ----------
        n : int
            Lattice dimension.
        steps : int
            Number of measurement steps or animation frames.
        p_S : float
            Probability of infection from suscetible.
        p_I : float
            Probability of recovered from infected.
        p_R : TYPE
            Probability of susceptible from recovered.

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the square lattice
        self.steps = steps # Measurement steps or animation frames 
        self.p_S = p_S # Probability of susceptible to infected
        self.p_I = p_I # Probability of infected to recovered
        self.p_R = p_R # Probability of recovered to susceptible
    
    def animate(self, steps):
        """
        Run and display an animation of the SIRS model spreading.

        Parameters
        ----------
        steps : int
            Total number of animation frames.

        Returns
        -------
        None.

        """
        
        # Initialise the lattice using the SIRS class
        sirs = SIRS(self.n, self.p_S, self.p_I, self.p_R)
        sirs.initialise()
        
        # Define the figure and axes for the animation
        fig, ax = plt.subplots()
        
        # Define custom cmap 
        # Use 4 colors: Grey (Vaccinated), Red (Infected), Black (Susceptible), Green (Recovered)
        sirs_cmap = ListedColormap(["#95a5a6", "#e74c3c", "#2c3e50", "#27ae60"])
        
        
        # Define custom cmap and colors
        # Values: -2 (Vaccinated), -1 (Infected), 0 (Susceptible), 1 (Recovered)
        colors = ["#95a5a6", "#e74c3c", "#2c3e50", "#27ae60"]
        labels = ["Vaccinated", "Infected", "Susceptible", "Recovered"]
        sirs_cmap = ListedColormap(colors)
        
        # Create legend handles manually
        # Use mpatches.Patch to create colored squares for the legend
        legend_handles = [mpatches.Patch(color=colors[i], label=labels[i]) 
                          for i in range(len(colors))]
        ax.legend(handles=legend_handles, loc='upper left', 
              bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.subplots_adjust(right=0.75)
        
        # Initialise the image object
        # vmin/vmax ensure -2 (vaccinated) is grey, -1 (infected) is red, 0 (susceptible) is black, 
        # and 1 (alive) is green consistently
        im = ax.imshow(sirs.lattice, cmap = sirs_cmap,
                       vmin = -2, vmax = 1)
        
        # Run the animation for the total number of steps
        for s in range(steps):
            
            # Update lattice
            sirs.update_lattice()
            
            # Update the animation 
            im.set_data(sirs.lattice)
            ax.set_title(f"Step: {s}")
            
            # Keep the image up while the script is running
            plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
    def calculate_average_infected(self, tot_infected_list):
        """
        Calculate the mean fraction of infected sites from a list of measurements.

        Parameters
        ----------
        tot_infected_list : list of int
            List containing counts of infected sites at different time steps.

        Returns
        -------
        float
            The average infected fraction <I>/N.

        """
        
        # Calculate and return the mean fraction of infected sites
        return np.mean(tot_infected_list) / self.N
    
    def calculate_variance_infected(self, tot_infected_list):
        """
        Calculate the normalised variance of the infected population.

        Parameters
        ----------
        tot_infected_list : list of int
            Counts of infected sites.

        Returns
        -------
        float
            The normalized variance: (<I^2> - <I>^2)/N.

        """
        
        # Convert to numpy array
        tot_infected_list = np.array(tot_infected_list)
        
        # Calculate and return the variance
        mean_infected_squared = np.mean(tot_infected_list**2)
        mean_infected = np.mean(tot_infected_list)
        return (mean_infected_squared - mean_infected**2) / self.N
    
    def bootstrap_method(self, data):
        """
        Estimate the standard error of the variance using the Bootstrap resampling method.

        Parameters
        ----------
        data : list or np.ndarray
            The population data to resample.

        Returns
        -------
        float
            Standard deviation of the resampled variances.

        """
        
        # Convert to numpy array
        data = np.array(data)
        
        # Find the length of the data
        n = len(data)
        
        # Create an empty list to store the resampled values
        resampled_values = []
        
        # Resampling 1000 times is sufficient
        for j in range(1000):
            
             # Randomnly resample from the n measurements
             ind = np.random.randint(0, n, size = n)
             resample = data[ind]
             
             # Calculate specific heat accordingly
             mean_E_sq = np.mean(resample**2)
             mean_E = np.mean(resample)
             value = (mean_E_sq - mean_E**2) / self.N
             
             # Append to the list
             resampled_values.append(value)
        
        # Calculate and return the error
        # Which is the standard deviation of the resampled values
        return np.std(np.array(resampled_values))
        
    def average_measurements(self, filename):
        """
        Sweep p_S and p_R to record the average infected fraction across the
        phase space.

        Parameters
        ----------
        filename : str
            Name of the output text file to save the results.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Define probabilities 
        p_I = 0.5
        p_S_array = np.round(np.arange(0, 1.05, 0.05), 2)
        p_R_array = np.round(np.arange(0, 1.05, 0.05), 2)
        
        # Create empty list to store results
        results = []
        
        # Iterate through both arrays of probabilities
        for p_S in p_S_array:
            for p_R in p_R_array:
                print(f"\rSimulating p_S={p_S}, p_R={p_R}...", end="", flush=True)
                
                # Initialise the lattice using the SIRS class
                sirs = SIRS(self.n, p_S, p_I, p_R)
                sirs.initialise()
                sirs.lattice = sirs.lattice
                
                # Make an empty list to hold at
                infected_list = []
                
                # Equilibrate for 100 sweeps
                for _ in range(100):
                    sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
                
                # Measure for 1000 sweeps
                for _ in range(1000):
                    sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
                    inf_count = np.count_nonzero(sirs.lattice == -1)
                    infected_list.append(inf_count)
                    
                    # Optimization: If the virus dies out, it stays dead
                    if inf_count == 0:
                        # Fill the rest of the list with 0 and break
                        remaining = 1000 - len(infected_list)
                        infected_list.extend([0] * remaining)
                        break
                    
                # After completing all the measurements for the probabilities
                # Calcaulte the average and the variance of the fraction of infected sites
                mean_infected = self.calculate_average_infected(infected_list)
                
                # Append a formatted string to results list
                results.append(f"{p_S},{p_R},{mean_infected}\n")
                
        # Write the values into the specified file
        with open(file_path, "w") as f:
            f.writelines(results)
                    
    def variance_measurements(self, filename):
        """
        Perform a sweep of p_S to measure variance and its error near phase transitions.

        Parameters
        ----------
        filename : str
            Output text file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Define probabilities 
        p_I = 0.5
        p_R = 0.5
        p_S_array = np.round(np.arange(0.2, 0.51, 0.01), 2)
        
        # Create empty list to hold results
        results = []
        
        # Iterate through the array of probabilities
        for p_S in p_S_array:
            print(f"\rSimulating p_S = {p_S} ...", end="", flush=True)
            
            # Initialise the lattice using the SIRS class
            sirs = SIRS(self.n, p_S, p_I, p_R)
            sirs.initialise()
            sirs.lattice = sirs.lattice
            
            # Make an empty list to hold data
            infected_list = []
            
            # Equilibrate for 100 sweeps
            for _ in range(100):
                sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
            
            # Measurements for 10000 sweeps
            for _ in range(10000):
                sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
                inf_count = np.count_nonzero(sirs.lattice == -1)
                infected_list.append(inf_count)
                
                if inf_count == 0:
                    # If infection dies, variance eventually becomes 0
                    remaining = 10000 - len(infected_list)
                    infected_list.extend([0] * remaining)
                    break
                
            # After completing all the measurements for the probabilities
            # Calcaulte the average and the variance of the fraction of infected sites
            variance_infected = self.calculate_variance_infected(infected_list)
            variance_infected_err = self.bootstrap_method(infected_list)
            
            # Append results to list
            results.append(f"{p_S},{p_R},{variance_infected},{variance_infected_err}\n")
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            f.writelines(results)
                
    def immunity_measurements(self, filename):
        """
        Measure the average infected fraction as a function of the vaccinated 
        fraction (immunity).

        Parameters
        ----------
        filename : str
            Output text file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Define probabilities 
        p_I = 0.5
        p_R = 0.5
        p_S = 0.5
         
        # Define array of fraction of vaccinated sites
        frac_immunity = np.round(np.linspace(0, 1, 101), 2) # 100 points
        
        # Create empty list to hold results 
        results = []
        
        # Iterate through fraction immunity array
        for frac in frac_immunity:
            print(f"\rSimulating f_Im = {frac}...", end="", flush=True)
               
            # Initialise the lattice using the SIRS class
            sirs = SIRS(self.n, p_S, p_I, p_R)
            sirs.initialise()
            sirs.lattice = sirs.lattice
            
            # Vaccinate the lattice
            sirs.vaccinate(frac)
            
            # Print how many sites are actually vaccinated
            #print(f"Number of vaccinated sites: {np.count_nonzero(sirs.lattice == -2)}")
            
            # Make an empty list to hold data
            infected_list = []
            
            # Equilibrate for 100 sweeps
            for _ in range(100):
                sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
            
            # Measure for 1000 sweeps
            for _ in range(1000):
                sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, p_S, p_I, p_R)
                inf_count = np.count_nonzero(sirs.lattice == -1)
                infected_list.append(inf_count)
                
                if inf_count == 0:
                    remaining = 1000 - len(infected_list)
                    infected_list.extend([0] * remaining)
                    break
                
            # After completing all the measurements for the probabilities
            # Calcaulte the average and the variance of the fraction of infected sites
            mean_infected = self.calculate_average_infected(infected_list)
            
            # Write results to results list
            results.append(f"{frac},{mean_infected}\n")
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            f.writelines(results)
                 
    def plot_immunity(self, filename):
        """
        Generate and save a plot of <I>/N versus the fraction of immune agents.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create an empty list to store input data
        input_data = []      
        
        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
        
        # Create empty lists to hold data 
        frac_immunity_array = []
        mean_infected_array = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain value from input data
            frac_immunity = float(input_data[i])
            mean_infected = float(input_data[i+1])

            # Append to empty lists
            frac_immunity_array.append(frac_immunity)
            mean_infected_array.append(mean_infected)
        
        # Plot variance with error bars using the bootstrap results
        ax.plot(frac_immunity_array, mean_infected_array, 'o-', 
                color='red', markerfacecolor='black', markeredgecolor='black',
                markersize=4, linewidth=1.5, label=r'$\langle I \rangle / N$')
        
        # Labels and formatting
        ax.set_xlabel(r"Fraction of Immune Agents $f_{Im}$", fontsize=14)
        ax.set_ylabel(r"Average Infected Fraction $\langle I \rangle / N$", fontsize=14)
        ax.set_title(r"SIRS Model: Effect of Vaccination ($p_S=p_I=p_R=0.5$)", fontsize=16)
        
        # Add a horizontal line at 0 to show when the infection is gone
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()

        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
                
    def plot_average_measurements(self, filename):
        """
        Generate and save a 2D phase diagram (heatmap) of average infection.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create an empty list to store input data
        input_data = []      
        
        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
        
        # Create empty lists to hold data
        p_S_array = []
        p_R_array = []
        mean_infected_array = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 3):
            
            # Obtain vlaue from input data
            p_S = float(input_data[i])
            p_R = float(input_data[i+1])
            mean_infected = float(input_data[i+2])

            # Append to empty lists
            p_S_array.append(p_S)
            p_R_array.append(p_R)
            mean_infected_array.append(mean_infected)
        
        # Get unique values to find grid dimensions
        # Figures out how many unique steps were taken
        unique_ps = np.unique(p_S_array)
        unique_pr = np.unique(p_R_array)
        
        # Reshape data into a 2D grid
        z_grid = np.array(mean_infected_array).reshape(len(unique_ps), len(unique_pr))

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot the 2D grid
        im = ax.imshow(z_grid, origin='lower', extent=[0, 1, 0, 1], 
                       aspect='auto', cmap='magma', interpolation='none')
        
        # Labels and formatting
        ax.set_title(r"Phase Diagram ($\langle I \rangle / N$)", fontsize=16)
        ax.set_xlabel(r"$p_{R \rightarrow S}$", fontsize=14)
        ax.set_ylabel(r"$p_{S \rightarrow I}$", fontsize=14)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        
        # Add a colorbar to quantify the average fraction of infected sites
        cbar = plt.colorbar(im)
        cbar.set_label(r"Average Infected Fraction $\langle I \rangle / N$", rotation=270, labelpad=20)
        
        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
            
    def plot_variance_measurements(self, filename):
        """
        Generate and save a plot of normalised variance with error bars.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create an empty list to store input data
        input_data = []      
        
        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
        
        # Create emptry lists to hold data 
        p_S_array = []
        variance_infected_array = []
        variance_infected_err_array = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 4):
            
            # Obtain vlaue from input data
            p_S = float(input_data[i])
            variance_infected = float(input_data[i+2])
            variance_infected_err = float(input_data[i+3])

            # Append to empty lists
            p_S_array.append(p_S)
            variance_infected_array.append(variance_infected)
            variance_infected_err_array.append(variance_infected_err)
        
        # Plot variance with error bars using the bootstrap results
        ax.errorbar(p_S_array, variance_infected_array, yerr=variance_infected_err_array, 
                    fmt='o-', color='red', ecolor='black', markerfacecolor = 'black', markeredgecolor = 'black',
                    capsize=3, elinewidth=1, markeredgewidth=1, markersize = 4, label = 'Infected Variance')
        
        # Labels and formatting to show the phase transition
        ax.set_xlabel(r"Infection Probability $p_{S \rightarrow I}$", fontsize=14)
        ax.set_ylabel(r"Normalised Variance $(\langle I^2 \rangle - \langle I \rangle^2)/N$", fontsize=14)
        ax.set_title(r"SIRS Variance Cut ($p_{R \rightarrow S} = 0.5$, $p_{I \rightarrow R} = 0.5$)", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellular Automata: SIRS")

    # User input parameters
    parser.add_argument("--n", type=int, default=50, help="Lattice size (n x n)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of measurement steps or animation frames.")
    parser.add_argument("--p_S", type=float, default=0.3, help="S -> I infection probability")
    parser.add_argument("--p_R", type=float, default=0.2, help="I -> R recovery probability")
    parser.add_argument("--p_I", type=float, default=0.5, help="R -> S immunity loss probability")
    parser.add_argument("--mode", type = str, default = "ani", 
                        choices = ["ani", "mea"],
                        help = "Animation or measurements")
    parser.add_argument("--measure", type = str, default = "heatmap", 
                         choices = ["average", "variance", "immunity"],
                         help = "Average or variance measurements")

    args = parser.parse_args()

    sim = Simulation(n=args.n, 
                     steps=args.steps,
                     p_S=args.p_S, 
                     p_R=args.p_R, 
                     p_I=args.p_I)
        
    if args.mode == "ani":
        
        sim.animate(steps = args.steps)
        
    if args.mode == "mea" and args.measure == "average":
        
        filename = "sirs_average_measurements.txt"
        sim.average_measurements(filename)
        sim.plot_average_measurements(filename)
        
    if args.mode == "mea" and args.measure == "immunity":
        
        filename = "sirs_immunity_measurements.txt"
        sim.immunity_measurements(filename)
        sim.plot_immunity(filename)
        
    if args.mode == "mea" and args.measure == "variance":
        
        filename = "sirs_variance_measurements.txt"
        sim.variance_measurements(filename)
        sim.plot_variance_measurements(filename)