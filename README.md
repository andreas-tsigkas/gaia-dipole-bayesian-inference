<h2 align="center">The Fellowship of the Dipole</h2>
<h3 align="center">A journey through Gaia astrometry and Bayesian inference.</h3>

---

## The Quest

> *“Even the smallest signal can change the course of the future.”*

This repository contains a research oriented Bayesian analysis of **real Gaia DR3 quasar
proper motion data**.

The goal is to identify and characterise **large scale patterns on the sky**, with a
focus on the dipole signal arising from the acceleration of the Solar System and global
rotation effects.

The methods are developed in the context of my MSc thesis in relativistic and
real time cosmology.

---

## The Road That Was Taken

In the full research code, the proper motion field is decomposed into **vector spherical
harmonics up to ℓ = 4**, resulting in a model with **48 global parameters**.

That full model exists elsewhere.

This repository deliberately presents a **minimal but faithful subset** of the analysis,
chosen so that the core ideas are readable, transparent, and reproducible.

---

## What Is Shown Here

### 1. The Gathering of the Data

An example **SQL query** is provided to show how Gaia DR3 quasar data are selected,
filtered, and joined across archive tables.

The query reflects the structure used in the full analysis. Credentials and direct
database access are intentionally omitted.

### 2. The Fellowship  
Dipole and Rotation

The central model implemented here is the **dipole plus rotation field**, often called
glide plus rotation.

It consists of **six parameters**:

- three dipole components ax ay az  
- three rotation components Rx Ry Rz  

Each source contributes a two dimensional proper motion measurement with a **full
covariance matrix**, incorporated analytically into a custom Gaussian likelihood.

Posterior inference is performed using PyMC.

### 3. Trials Along the Way

The analysis explores how the inferred dipole parameters depend on **redshift**.

The data are divided into bins and the model is refit independently, mirroring the
strategy used in the full thesis analysis.

---

## The Path of the Analysis

> *“One does not simply fit a model.”*

The workflow follows a clear sequence:

- select Gaia sources using SQL  
- implement a physically interpretable proper motion field  
- build a custom likelihood with correlated uncertainties  
- perform Bayesian inference  
- compare results across subsamples  

The emphasis is on **statistical modelling and uncertainty handling**, not on software
abstraction.

---

## The Tools of the Fellowship

- Python  
- NumPy and Pandas  
- [PyMC](https://www.pymc.io) and PyTensor  
- ArviZ  
- SQL via the Gaia Archive  

---

## For Those Who Walk Different Roads

This repository can be read in two ways.

**For academic readers**  
It demonstrates Bayesian modelling of astrometric data, vector field inference on the
sphere, and uncertainty aware parameter estimation.

**For industry and data science readers**  
It demonstrates SQL based data selection, construction of custom likelihoods, handling
correlated uncertainties, and reproducible statistical analysis in Python.

---

> *“There are older and fouler things than Orcs in the deep places of the sky.”*

Large scale astrometric effects require careful modelling and comparisons
across subsamples.

Measurement uncertainties are incorporated analytically in the likelihood.
