# LeakG3PD-OD: a Dataset for Water Distribution System Leakage Detection and Localization Using Water Demand Automated Meter Reading

This dataset was based on [LeakG3PD EPANET Net 3 dataset](https://github.com/matheuspilotto/LeakG3PD), which was considered as representative of the Real System.
The objective was to evaluate the use of almost exclusively demand signals for Leakage Detection and Localization.
In order to do that:
- node demand values were copied (as if measured) from Real System and preprocessed including missing values, frauds and measurement errors
- in the original System Model (the one used as basis to create the 500 LeakG3PD Epanet Net 3 scenarios):
	- preprocessed demands were attributed to nodes
	- tanks were replaced by reservoirs
	- pumps were replaced by negligible roughness pipes with the same open/close controls from pumps
	- all reservoirs (including the tank substitutes) were attributed head patterns in order to reproduce pressures measured from Real System
	- original Real System pressures and link flows .csv files were deleted
	- simulation was run for all scenarios
	- System Model estimated pressures and estimated link flows .csv files were saved
	- once leak nodes don't exist in System Model, they were referred by their distance to leak link start nodes and such information was saved in "Leaks" folder
	- such dataset was named LeakG3PD-OD

<a href="https://drive.google.com/file/d/1pqXk8KCyxDKMRbDs1uR1cnIr1snOya3f/view?usp=sharing"><img src="https://drive.google.com/file/d/1pqXk8KCyxDKMRbDs1uR1cnIr1snOya3f/view?usp=sharing" width="600" height="240"/><a>

<a href="https://drive.google.com/file/d/1UgqVStuzCb4xDHcqf9BGglu763M_w9D8/view?usp=sharing"><img src="https://drive.google.com/file/d/1UgqVStuzCb4xDHcqf9BGglu763M_w9D8/view?usp=sharing" width="600" height="240"/><a>

<a href="https://drive.google.com/file/d/1L2ovbDeq5ToxIl7rnxhcMLqZKxZUJtXg/view?usp=sharing"><img src="https://drive.google.com/file/d/1L2ovbDeq5ToxIl7rnxhcMLqZKxZUJtXg/view?usp=sharing" width="600" height="240"/><a>

Using LeakG3PD-OD dataset, leak detection and localization algorithms were developed.

<a href="https://drive.google.com/file/d/1pqXk8KCyxDKMRbDs1uR1cnIr1snOya3f/view?usp=drive_link"><img src="https://drive.google.com/file/d/1pqXk8KCyxDKMRbDs1uR1cnIr1snOya3f/view?usp=drive_link" width="600" height="240"/><a>

<a href="https://drive.google.com/file/d/1SD-5jUqxTUs1A_VNJLaM6RjYL-eGKpsb/view?usp=sharing"><img src="https://drive.google.com/file/d/1SD-5jUqxTUs1A_VNJLaM6RjYL-eGKpsb/view?usp=sharing" width="600" height="240"/><a>

The results showed that both algorithms are useful, but some adjusts are necessary in leak detection functional parameters and that LeakG3PD pipe parameter uncertainties are excessive reducing the localization performance.	

# Instructions for reproducibility

In any case, to run Python programs:
1. Download Anaconda3
2. Install VS Code
3. conda install conda-forge::wntr

If you wish to generate LeakG3PD-OD from scratch:
4. Download and unzip [LeakG3PD Epanet Net 3 dataset](https://drive.google.com/drive/folders/1HM2xI9VpC4us7rFX4IuXXCoHDrnWfC17?usp=sharing) folder into a "LeakG3PD-OD Evaluation" folder
5. Run python source file ODDatasetGenerator.py
6. Rename the dataset folder as EPANET Net 3_OD

If you just wish to run the LDL program:
7. Download and unzip  [LeakG3PD-OD Epanet Net 3 dataset](https://drive.google.com/drive/folders/11qlW5CKUmyL0-IXvqJ7_YtzUEqelhG63?usp=sharing)
8. Run python source file ODLeakDetectionAndLocalization.py