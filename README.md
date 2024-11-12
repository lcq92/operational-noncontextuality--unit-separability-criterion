(Project from 2021, implementing the *Unit Separability criterion* algorithm proposed in https://quantum-journal.org/papers/q-2022-06-07-732/, to assess whether a quantum prepare-and-measure scenario is *operationally-noncontextual* )

## Requirements

- This project is tested to run using 
    - Python 3.8.8
    - pplpy 0.8.6

- The simplest way to get a working environment to launch this in, is to install Conda and then run: 		

	```bash
	conda create -n classicality python=3.8.8 jupyter numpy scipy matplotlib

	conda activate classicality

	conda install -c conda-forge jupyterlab tqdm

	conda install -c conda-forge pplpy
	```
	(this last command will take care of installing all of `pplpy`'s dependencies)
