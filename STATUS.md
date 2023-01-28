# Badges

We apply for the three badges *available*, *functional*, and *reusable*. Here are the reasons why we believe that the artifact deserves the badges.

### Available
The artifact is publicly shared in the GitHub repository: `https://github.com/sumonbis/Fairify`.
Furthermore, the latest release is published in Zenodo and a DOI is obtained:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7579939.svg)](https://doi.org/10.5281/zenodo.7579939)

### Functional and Reusable
The artifact contains both the data and software components that is executable. We also automated running the tool using command line interface. Specifically, our artifact contains the following to be fully functional. 

1. All three datasets needed to run the tool. 
2. The benchmark models packaged in `h5` files for analysis. 
    a. We collected models from open source Kaggle repositories. The curated models are included. 
    b. We collected benchmark models from three prior works which are also included in the artifact. 
3. We shared executable Python source code to run the tool.

We also provided detailed instructions for installation, environment setup, and running the tool. Furthermore, we shared the following necessary components of the artifact to make it reusable:

* Instructions to create virtual Python environment and run the tool in that environment. This will avoid modifying the users' home environment setup. 
* We provided general instructions to run the tool on any operating system. 
* Automatically install necessary packages.
* Provide bash files to automate the pipeline.
* The experimental results are shared in the CSV files, which are used in the paper.
