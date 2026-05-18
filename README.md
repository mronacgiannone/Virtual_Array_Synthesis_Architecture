# Virtual_Array_Synthesis_Architecture (VASA)

Repository associated with the manuscript, "Learning Seismic Wavefield Structure from Regional Arrays with Self-Supervised Deep Learning" (in review).

## Notebooks

- 1a_Prelim_Catalog_Analysis.ipynb
  - Explores the regional and local earthquake catalogs and computes event azimuth and distance relative to the PFO array.
- 1b_Database_Construction.ipynb
  - Performs waveform quality control and constructs the final event database used for VASA
- 2_Data_Preprocessing.ipynb
  - Converts the compiled waveform database into the final model-ready dataset by filtering the waveforms, removing poorly reconstructed low-magnitude events, evaluating spatial coherency, and producing the train/test split and associated metadata used for VASA

## Folders

- Regional_Catalog
  - Stores the catalog query outputs and associated search settings for events surrounding the PFO array. The folder includes the local, intermediate, and regional catalog searches used to characterize the broader seismicity distribution and source-receiver geometry
- Local_Catalog
  - Curated local event catalog for the PFO array, including the final catalog file and quality-control spreadsheets listing events removed automatically and manually during database construction.

