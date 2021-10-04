# Roman-CGI-POPPY

This code is archived and can be cited via Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5112382.svg)](https://doi.org/10.5281/zenodo.5112382)

For more details see:

Milani, K., Douglas, E. S., & Ashcraft, J. (2021). Updated simulation tools for Roman coronagraph PSFs. In UV/Optical/IR Space Telescopes and Instruments: Innovative Technologies and Concepts X (Vol. 11819, p. 118190E). International Society for Optics and Photonics. https://arxiv.org/abs/2108.10924

This repository is the development repo utilized for updating Roman CGI simulation tools by translating the wfirst_phaseb_proper models to POPPY. 

To begin with, all the data files required can be found on Box at this [link](https://arizona.box.com/s/9cquzldre5ru2497a70omstxb1o83l5s). 

The POPPY models are defined in the Python file poppy_romancgi_modes.py. The file used to run the modes is the jupyter notebook run_poppy_modes.ipynb. In this notebook, the poppy_romancgi_modes file is imported and a PSF is calculated with the function run_model(). There are various kwargs allowing for different options to be utilized when calculating a PSF, but overall, these models are only intedned for monochromatic PSF calculations. 

The polmap.py file is what is used to implement the polarization aberrations. The file is originally from the wfirst_phaseb_proper package, but some alterations were made in order to function with a FresnelWavefront from POPPY so that it could be used as the input wavefront when calculating a PSF. 

The misc.py file contains some useful plotting tools for showing the wavefront results and a function to either pad or crop a square 2D array. 

When it comes to utilizing the POPPY accelerated math options, the FFT types used here were pyFFTW, mkl_fft, and PyOpenCL, all of which were installed through conda-forge. The packages pyFFTW and mkl_fft were usable immediately after install, but the PyOpenCL package was trickier to use as it required other packages to be installed. Another package required for PyOpenCL was ocl-icd-system, which was also installed through conda-forge, as PyOpenCl needed to recignize the GPU in the system. For the case of other machines, a different package from ocl-icd-system may be needed, in which case, the [PyOpenCL instructions website](https://documen.tician.de/pyopencl/misc.html) may be useful. Lastly, POPPY aso requires gpyfft for the PyOpenCL FFTs. This package was downloaded from the [gpyfft GitHub](https://github.com/geggo/gpyfft) page and installed through the command line by running the setup.py file after adding some CL header files to the gpyfft directory. 

The wfirst_phaseb_proper models are run by using the jupyter notebook run_romancgi_proper. To run these models, the wfirst_phaseb_proper package should be installed along with PROPER itself. The versions used were wfirst_phaseb_proper V1.7 and PROPER V3.2.3.  

Along with the notebook to run the PROPER models, some previous results from PROPER were saved into the proper-psfs directory such that comparison could be done more quickly. 

On a side note, it may be useful to know how to install the latest version of POPPY directly from the GitHub repository. The following command is what was used to do so. The -e option is not required, but it will install POPPY in a directory called src and makes editing POPPY a bit easier, as you do not have to navigate to the site-packages directory of the python/conda environment whenever viewing or editing POPPY's source code. 

pip install -e git+https://github.com/spacetelescope/poppy.git#egg=poppy 

