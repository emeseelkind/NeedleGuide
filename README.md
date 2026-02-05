# KidneyNav: Real-time kidney ultrasound segmentation and 3D reconstruction in 3D Slicer

### Authors

- Gabriella d'Albenzio (Queen's University, Canada)
- Kyle Sunderland (Queen's University, Canada)
- Emese Elkind (Queen's University, Canada)
- Lily Morrell (Queen's University, Canada)
- Ron Kikinis (BWH, USA)
- Gabor Fichtinger (Queen's University, Canada)

### Project Description
KidneyNav is a 3D Slicer scripted module designed for real-time ultrasound navigation and intraoperative visualization. The module connects to a PLUS server via OpenIGTLink to stream live 2D ultrasound images and tracking transforms, and it supports live volume reconstruction using Slicer’s VolumeReconstruction infrastructure. The current implementation includes automatic node setup (input image, prediction volume, transforms, connectors, reconstruction node/ROI), a custom 2D/3D layout for simultaneous slice and 3D rendering, and tools to record synchronized sequences (ultrasound, predictions, transforms, and needle model) for later analysis.

During this Project Week, we want to validate the module in a real live scanning setting and integrate real-time AI-based multiclass segmentation (kidney + calyces + fluid)). We also want to connect multiclass predictions to live volume reconstruction and discuss best practices for reconstructing volumes from two different (or complementary) prediction streams (e.g., kidney mask vs calyces mask, or two model outputs), including visualization, synchronization, and reconstruction strategies.

### Objective
1. Objective A. Validate the end-to-end live workflow by recording synchronized ultrasound, prediction, transform, and needle model sequences during real-time acquisition.

2. Objective B. Integrate real-time multiclass AI segmentation (kidney, calyx, fluid) streamed into 3D Slicer via OpenIGTLink and used directly for live volume reconstruction.

3. Objective C. Establish and compare reconstruction strategies for multiclass predictions, including single-volume and dual-volume approaches, and derive community-informed recommendations on synchronization, labeling, interpolation, and fusion.



### Installation
To use this extension, 3D Slicer should be downloaded [here](https://download.slicer.org/) as well as  Anaconda/Miniconda installation instructions can be found [here](https://www.anaconda.com/download).

Create a Conda environment from the environment.yml by following [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) or within this repository's root directory the following:

````conda env create -f environment.yml````

Optionally, you can add the flag ````-n```` followed by the name that you want your environment to be.


### Running the Module
This module should be used on an ultrasound scene of the kidney.

1.  Download this repository and save it to your local computer.  
2. Locate the module file here:  
   **[`KidneyNav/KidneyNav.py`](KidneyNav/KidneyNav.py)**  
3. Open **3D Slicer**.  
4. Drag and drop `KidneyNav.py` into Edit → Application Settings → Modules → Additional Module Paths directory of 3D Slicer.
   - You will be asked to restart 3D Slicer
5. After restarting, search for **KidneyNav** in the Modules list and open it.
  
 
### Approach and Plan
1. Record short test sequences (ultrasound image, prediction, transforms, needle model).

2. Stream multiclass prediction (kidney / calyx / fluid) into Slicer via OpenIGTLink as a label or scalar volume synchronized with the ultrasound stream.

3. Run live volume reconstruction directly from the prediction stream.

4. Review with the Slicer/IGT community best practices for synchronization, label consistency, interpolation, and reconstruction from dual prediction outputs.

5. Compare reconstruction strategies:
   - Creates and registers a volume rendering preset that visualizes segmentation labelmaps using a discrete transfer function derived from Segmentation_ColorTable.
   - Visualizes multi-class predictions as a multi-component scalar volume, using Slicer’s Independent Multi-Component Volume Rendering mode
