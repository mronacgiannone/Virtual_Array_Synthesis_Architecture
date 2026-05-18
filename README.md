# Virtual_Array_Synthesis_Architecture (VASA)

Repository associated with the manuscript, "Learning Seismic Wavefield Structure from Regional Arrays with Self-Supervised Deep Learning" (in review).

## Notebooks

- 1a_Prelim_Catalog_Analysis.ipynb
  - Explores the regional and local earthquake catalogs and computes event azimuth and distance relative to the PFO array.
- 1b_Database_Construction.ipynb
  - Performs waveform quality control and constructs the final event database used for VASA
- 1c_Spatial_Coherence.ipynb
  - Evaluates spatial coherence and source-geometry coverage of the resulting array dataset
- 2_Data_Preprocessing.ipynb
  - Converts the compiled waveform database into the final model-ready dataset by filtering the waveforms, removing poorly reconstructed low-magnitude events, evaluating spatial coherence, and producing the train/test split and associated metadata used for VASA

