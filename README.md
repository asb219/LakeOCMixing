# RICH-sediments

Isotopic Bayesian mixing model for organic carbon in lake sediments.
Case study: Lake Constance Obersee.

Sources of organic carbon (OC) in lake sediments:

* Autochthonous OC: produced through lake primary productivity
* Petrogenic OC: bedrock-derived OC
* Pedogenic OC: soil-derived OC

Observational data in sediment core:

* Δ<sup>14</sup>C: radiocarbon signature of sediment OC
* δ<sup>13</sup>C: stable isotope signature of sediment OC
* TOC/TN: mass ratio of total organic carbon to total nitrogen
* ROC/TOC: proportion of residual oxidizable carbon in sediment OC

The mixing model results for a single core are then extrapolated
over the entire Lake Constance Obersee using ordinary Kriging,
with depth and distance from the Rhine inflow as external drift parameters.


**Associated manuscript**:
Mittelbach et al. (_in prep._).
"Pre-aged organic matter dominates modern organic carbon burial in a major perialpine lake system".
_Limnology and Oceanography_.


## Contents

* `mcmc.py`: Calibrate mixing model on a sediment core with MCMC using the `pymc` package.
* `Kriging.ipynb`: Perform Kriging over the entire lake using the `pykrige` package.
* `Figures.ipynb`: Produce figures for the manuscript.
* `environment.yml`: Package requirements for conda environment.


## Getting started

### Clone the repository

Clone this repository with
```
git clone https://github.com/asb219/RICH-sediments.git
```

Now the repository should be in a directory named `RICH-sediments`.
Move into that directory with `cd RICH-sediments`.

### Create the virtual environment with conda

Create and activate this project's conda environment (called `sedmix` by default):
```
conda env create -f environment.yml
conda activate sedmix
```

### Run the code

Once the virtual environment is activated, you can run the MCMC calibration
of the mixing model with `python mcmc.py`.

To perform Kriging over Lake Constance and reproduce the plots in the manuscript,
launch jupyter with `jupyter notebook` and run the `Kriging.ipynb`
and `Figures.ipynb` notebooks, respectively.

All the resulting data files and figures will be written inside the `output` directory.


## Issues

If you encounter a bug or other issues, please raise an issue
at https://github.com/asb219/RICH-sediments/issues.

Feel free to fork this repository and implement your own changes and features.
