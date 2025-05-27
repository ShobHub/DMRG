# DMRG

# Quantum Dimer Model – DMRG Simulation Code

This repository contains the complete DMRG codebase used to study the Quantum Dimer Model (QDM), with hard dimer constraints explicitly encoded. 

## Overview

The code implements the Density Matrix Renormalization Group (DMRG) technique and supporting analysis tools to explore the ground state properties and phase transitions in the QDM.

## Files and Descriptions

- **QDM_Initialisation_AB.py**  
  Main script to initialize and run DMRG simulations for the QDM.

- **Lanczos_HV.py**  
  Implements the Lanczos algorithm for two-site DMRG optimisation, used to compute eigenvalues and eigenvectors of the Hamiltonian.

- **DimerDensity.py**  
  Calculates onsite basis probabilities to characterise the structure of the ground state across different parameter regimes.

- **PhaseT.py**  
  Analyses the nature of phase transitions by evaluating order parameters and tracking energy variations.
