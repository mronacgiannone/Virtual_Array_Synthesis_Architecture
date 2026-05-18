# Virtual Array Synthesis Architecture (VASA)

## Repository Overview

Repository associated with the manuscript, "Learning Seismic Wavefield Structure from Regional Arrays with Self-Supervised Deep Learning" (in review).

## Notebooks

- **1a_Prelim_Catalog_Analysis.ipynb**
  - Explores the regional and local earthquake catalogs and computes event azimuth and distance relative to the PFO array.
- **1b_Database_Construction.ipynb**
  - Performs waveform quality control and constructs the final event database used for VASA.
- **2_Data_Preprocessing.ipynb**
  - Converts the compiled waveform database into the final model-ready dataset by filtering the waveforms, removing poorly reconstructed low-magnitude events, evaluating spatial coherency, and producing the train/test split and associated metadata used for VASA.
- **3a_Visualize_VASA.ipynb**
  - Displays the full VASA architecture for inspection and documentation.
- **3b_Train_VASA_v1.ipynb**
  - Implements training of the initial VASA architecture using the finalized train/test split and provides an initial evaluation of model behavior, including validation tracking and qualitative reconstruction checks.
- **3c_Evaluate_VASA_v1.ipynb**
  - Applies FK array processing to observed and reconstructed waveforms in order to evaluate whether VASA preserves the directional and kinematic structure of the seismic wavefield during sensor reconstruction.

## Folders

- **Regional_Catalog/**
  - Stores the catalog query outputs and associated search settings for events surrounding the PFO array. The folder includes the local, intermediate, and regional catalog searches used to characterize the broader seismicity distribution and source-receiver geometry.
- **Local_Catalog/**
  - Curated local event catalog for the PFO array, including the final catalog file and quality-control spreadsheets listing events removed automatically and manually during database construction.
- **Database/**
  - **waveforms.npy** (Zenodo): Disk-backed NumPy array containing the compiled waveform database. Each event is stored as a fixed-size tensor of filtered three-component waveforms across the retained PFO stations, providing the core input data used for subsequent preprocessing and model training. (Notebook 1b)
  - **event_meta.pkl**: Pickled pandas Dataframe containing the event-level metadata associated with waveforms.npy. Each row corresponds to one waveform entry in the database and includes the catalog information and source file path needed to track each event through preprocessing and analysis. (Notebook 1b)
  - **dataset_info.pkl**: Pickled summary dictionary describing the constructed waveform database. This file stores key structural metadata such as array shape, data type, component order, station order, target waveform length, and bookkeeping lists for missing or failed events during database assembly. (Notebook 1b)
  - **Coherency_Results/**
    - **corr_df.pkl**: Pickled Dataframe containing event-level or summary coherency results used in the single-band and multi-band coherency analyses. (Notebook 2)
    - **curve_df.pkl**: Pickled DataFrame containing the coherence curves or aggregated coherence-versus-distance results used for downstream plotting and inspection. (Notebook 2)
    - **corr_df.csv**: CSV version of **corr_df.pkl** for quick inspection outside Python. (Notebook 2)
    - **curve_df.csv**: CSV version of **curve_df.pkl** for quick inspection outside Python. (Notebook 2)
    - **pair_corr_df.pkl**: Pickled Dataframe containing pairwise inter-station cross-correlation measurements, used for the station-spacing coherency analysis. (Notebook 2)
    - **pair_corr_df.csv**: CSV version of **pair_corr_df.pkl**. (Notebook 2)
  - **Preprocessed/**
    - **X_Train5.npy** (Zenodo): Preprocessed training waveform array bandpass filtered 0.5 - 5 Hz fpr VASA model development. (Notebook 2)
    - **X_Test5.npy** (Zenodo): Preprocessed testing waveform array bandpass filtered 0.5 - 5 Hz fpr VASA model development. (Notebook 2)
    - **X_Traing10.npy** (Zenodo): Preprocessed training waveform array bandpass filtered 0.5 - 10 Hz fpr VASA model development. (Notebook 2)
    - **X_Test10.npy** (Zenodo): Preprocessed testing waveform array bandpass filtered 0.5 - 10 Hz fpr VASA model development. (Notebook 2)
    - **meta_train.pkl**: Pickled DataFrame containing the event metadata corresponding to the train split waveforms
    - **meta_test.pkl**: Pickled DataFrame containing the event metadata corresponding to the test split waveforms
