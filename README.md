# Virtual Array Synthesis Architecture (VASA)

## Repository Overview

This repository provides the code and analysis workflow for building, training, and evaluating VASA (Virtual Array Synthesis Architecture). The project is centered on reconstructing missing seismic array channels from the remaining observed sensors using a convolutional-Transformer hybrid model.

The repository includes:

- catalog preparation and event selection
- waveform database construction and preprocessing
- model architecture visualization and training
- FK-based physical validation of reconstructed wavefields
- interpretability analyses of sensor importance

IMPORTANT: All files needed to reproduce the results within the study can be found either here or on the Zenodo repository. The only thing missing are the .mseed files with instrument response removed (not enough storage on Zenodo). These waveforms were bandpass filtered, demeaned, and detrended (and certain events were removed due to data unavailability). The resulting waveforms.npy, event_meta.pkl, and dataset_info.pkl are accessible and were created in notebook 1b. So please begin processing from notebook 2. If you would like the original .mseed files to run notebooks 1a and 1b, please contact corresponding author at miroronac24@gmail.com.

## Cardinal

The Cardinal software package must be installed prior to environment set up. Information on installing Cardinal can be found here: https://github.com/sjarrowsmith/cardinal.git

## Install and Activate

Navigate to directory and type:

 - conda env create -f deep_learning_env.yml
 - source activate deep_learning
 - pip install tensorflow==2.18.0 keras==3.8.0
 - conda install graphviz
 - pip install pydot dask cartopy networkx future pisces

## Notebooks

- **1a_Prelim_Catalog_Analysis.ipynb**
  - Explores the regional and local earthquake catalogs and computes event azimuth and distance relative to the PFO array.
- **1b_Database_Construction.ipynb**
  - Performs waveform quality control and constructs the final event database used for VASA (waveforms.npy, event_meta.pkl, and dataset_info.pkl - these can be found on the corresponding Zenodo repository)
- **2_Data_Preprocessing.ipynb**
  - Converts the compiled waveform database into the final model-ready dataset by filtering the waveforms, removing low-magnitude events to reach dataset size of 30,000, evaluating spatial coherency, and producing the train/test split and associated metadata used for VASA. This is where the files meta_test.pkl, meta_train.pkl, X_test5.npy, X_test10.npy, X_train5.npy, and X_train10.npy are all constructed (these files can be found on the Zenodo repository).
- **3a_Visualize_VASA.ipynb**
  - Displays the full VASA architecture for inspection and documentation.
- **3b_Train_VASA_v1.ipynb**
  - Implements training of the initial VASA architecture using the finalized train/test split and provides an initial evaluation of model behavior, including validation tracking and qualitative reconstruction checks. This is where the files Database/Preprocessed/global_component_scaling.npz, Models/vasa_v1.keras, Models/vasa_v1_1000epochs.keras, Models/vasa_v1_10Hz.keras/ Models/vasa_v1_10Hz_1000epochs.keras, Training_Logs/training_log_v1.csv, and Training_Logs/training_log_v1_10Hz.csv are constructed. This is also where the attention analysis was performed and where the sensor removal experiments were conducted.
- **3c_Evaluate_VASA_v1.ipynb**
  - Applies FK array processing to the fully observed array, reconstructed array, and reduced array. This was also where VASA performance vs the three spatial interpolation baselines were conducted. The robustness examples (multi-event arrivals, localized noise, coherent transients, and temporal offsets) were also produced here. Finally, the component ablation experiments were also conducted inside this notebook.

## Folders

- **Regional_Catalog/**
  - Stores the catalog query outputs and associated search settings for events surrounding the PFO array. The folder includes the local, intermediate, and regional catalog searches used to characterize the broader seismicity distribution and source-receiver geometry.
- **Local_Catalog/**
  - Curated local event catalog for the PFO array, including the final catalog file and quality-control spreadsheets listing events removed automatically and manually during database construction.
- **Training_Logs/**
  - Contain the training results for the 0.5 - 5 Hz and the 0.5 - 10 Hz models
- **Models/** (create this directory - the files that need to be in here can be found in the Zenodo repository)
  - **vasa_v1.keras** (Zenodo): weights for the 0.5 - 5 Hz model after training for 500 epochs
  - **vasa_v1_1000epochs.keras** (Zenodo): weights for the 0.5 - 5 Hz model after training for 1000 epochs
  - **vasa_v1_10Hz.keras** (Zenodo): weights for the 0.5 - 10 Hz model after training for 500 epochs
  - **vasa_v1_10Hz_1000epochs.keras** (Zenodo): weights for the 0.5 - 10 Hz model after training for 1000 epochs 
- **Database/** (create this directory - the files that need to be in here can be found in the Zenodo repository)
  - **waveforms.npy** (Zenodo): Disk-backed NumPy array containing the compiled waveform database. Each event is stored as a fixed-size tensor of filtered three-component waveforms across the retained PFO stations, providing the core input data used for subsequent preprocessing and model training. (Notebook 1b)
  - **event_meta.pkl** (Zenodo): Pickled pandas Dataframe containing the event-level metadata associated with waveforms.npy. Each row corresponds to one waveform entry in the database and includes the catalog information and source file path needed to track each event through preprocessing and analysis. (Notebook 1b)
  - **dataset_info.pkl** (Zenodo): Pickled summary dictionary describing the constructed waveform database. This file stores key structural metadata such as array shape, data type, component order, station order, target waveform length, and bookkeeping lists for missing or failed events during database assembly. (Notebook 1b)
  - **Coherency_Results/** (create this subdirectory within Database - the files that need to be in here can be found in the Zenodo repository)
    - **corr_df.pkl** (Zenodo): Pickled Dataframe containing event-level or summary coherency results used in the single-band and multi-band coherency analyses. (Notebook 2)
    - **curve_df.pkl** (Zenodo): Pickled DataFrame containing the coherence curves or aggregated coherence-versus-distance results used for downstream plotting and inspection. (Notebook 2)
    - **corr_df.csv** (Zenodo): CSV version of **corr_df.pkl** for quick inspection outside Python. (Notebook 2)
    - **curve_df.csv** (Zenodo): CSV version of **curve_df.pkl** for quick inspection outside Python. (Notebook 2)
    - **pair_corr_df.pkl** (Zenodo): Pickled Dataframe containing pairwise inter-station cross-correlation measurements, used for the station-spacing coherency analysis. (Notebook 2)
    - **pair_corr_df.csv** (Zenodo): CSV version of **pair_corr_df.pkl**. (Notebook 2)
  - **Preprocessed/** (create this directory - the files that need to be in here can be found in the Zenodo repository)
    - **X_train5.npy** (Zenodo): Preprocessed training waveform array bandpass filtered 0.5 - 5 Hz fpr VASA model development. (Notebook 2)
    - **X_test5.npy** (Zenodo): Preprocessed testing waveform array bandpass filtered 0.5 - 5 Hz fpr VASA model development. (Notebook 2)
    - **X_train10.npy** (Zenodo): Preprocessed training waveform array bandpass filtered 0.5 - 10 Hz fpr VASA model development. (Notebook 2)
    - **X_test10.npy** (Zenodo): Preprocessed testing waveform array bandpass filtered 0.5 - 10 Hz fpr VASA model development. (Notebook 2)
    - **meta_train.pkl** (Zenodo): Pickled DataFrame containing the event metadata corresponding to the train split waveforms
    - **meta_test.pkl** (Zenodo): Pickled DataFrame containing the event metadata corresponding to the test split waveforms
    - **global_component_scaling.npz** (Zenodo): global center and scale to scale dataset
