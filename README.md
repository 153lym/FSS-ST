# FSS-ST

FSS-ST: Dynamically Capturing Reliable Distant Semantics for Identifying Fine-Grained Spatial Domain

## Overview:

We introduce FSS-ST.


## Run environment:
 
FSS-ST is implemented in the pytorch framework. The detailed running environment can be found in file [FSS-ST.yaml](FSS-ST.yaml).
In the experiments, the GPU we used was NVIDIA RTX A6000.

## Run

### ST raw dataset download
The mouse & human liver dataset with 10x Visium platform is collected from https://www.livercellatlas.org/. 

The HER2 Positive Breast Tumors dataset with ST platform is collected from https://github.com/almaan/her2st. 

The DLPFC dataset with 10x Visium platform is accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD). 

The mouse medial prefrontal cortex  dataset with STARmap platform is collected from http://clarityresourcecenter.org/. 

The MOSTA hypothalamic preoptic region with Stereo platform is collected from https://db.cngb.org/stomics/mosta/. 

### Run FSS-ST

We give example on the mouse liver datasets with 10x Visium platform. 
We provide the example code in [main_mouse.py](main_mouse.py).

`python main_mouse.py`

For different ST data, it is necessary to set the range of alpha and gamma parameters to find the optimal set of parameter values. 
During the experiment, we set the range of [5, 10] for both.

## Citation:
**This repository contains the source code for the paper:**

`FSS-ST: Dynamically Capturing Reliable Distant Semantics for Identifying Fine-Grained Spatial Domain`
