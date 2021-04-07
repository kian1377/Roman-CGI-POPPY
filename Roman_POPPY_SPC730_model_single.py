import astropy.io.fits as fits
import astropy.units as u
import scipy
import poppy
from poppy.poppy_core import PlaneType
import proper
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from IPython.display import display, clear_output

import os
import time
from pathlib import Path

import logging, sys
_log = logging.getLogger('poppy')
_log.setLevel("DEBUG")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from importlib import reload
import misc

fl_pri = 2.83459423440 * 1.0013*u.m
d_pri_sec = 2.285150515460035*u.m
d_focus_sec = d_pri_sec - fl_pri
fl_sec = -0.653933011 * 1.0004095*u.m
d_sec_focus = 3.580188916677103*u.m
diam_sec = 0.58166*u.m
d_sec_fold1 = 2.993753476654728*u.m
d_fold1_focus = 0.586435440022375*u.m
diam_fold1 = 0.09*u.m
d_fold1_m3 = 1.680935841598811*u.m
fl_m3 = 0.430216463069001*u.m
d_focus_m3 = 1.094500401576436*u.m
d_m3_pupil = 0.469156807701977*u.m
d_m3_focus = 0.708841602661368*u.m
diam_m3 = 0.2*u.m
d_m3_m4 = 0.943514749358944*u.m
fl_m4 = 0.116239114833590*u.m
d_focus_m4 = 0.234673014520402*u.m
d_m4_pupil = 0.474357941656967*u.m
d_m4_focus = 0.230324117970585*u.m
diam_m4 = 0.07*u.m
d_m4_m5 = 0.429145636743193*u.m
d_m5_focus = 0.198821518772608*u.m
fl_m5 = 0.198821518772608*u.m
d_m5_pupil = 0.716529242882632*u.m
diam_m5 = 0.07*u.m
d_m5_fold2 = 0.351125431220770*u.m
diam_fold2 = 0.06*u.m
d_fold2_fsm = 0.365403811661862*u.m
d_fsm_oap1 = 0.354826767220001*u.m
fl_oap1 = 0.503331895563883*u.m
diam_oap1 = 0.06*u.m
d_oap1_focm = 0.768005607094041*u.m
d_focm_oap2 = 0.314483210543378*u.m
fl_oap2 = 0.579156922073536*u.m
diam_oap2 = 0.06*u.m
d_oap2_dm1 = 0.775775726154228*u.m
d_dm1_dm2 = 1.0*u.m
d_dm2_oap3 = 0.394833855161549*u.m
fl_oap3 = 1.217276467668519*u.m
diam_oap3 = 0.06*u.m
d_oap3_fold3 = 0.505329955078121*u.m
diam_fold3 = 0.06*u.m
d_fold3_oap4 = 1.158897671642761*u.m
fl_oap4 = 0.446951159052363*u.m
diam_oap4 = 0.06*u.m
d_oap4_pupilmask = 0.423013568764728*u.m
d_pupilmask_oap5 = 0.408810648253099*u.m
fl_oap5 =  0.548189351937178*u.m
diam_oap5 = 0.06*u.m
d_oap5_fpm = 0.548189083164429*u.m
d_fpm_oap6 = 0.548189083164429*u.m
fl_oap6 = 0.548189083164429*u.m
diam_oap6 = 0.06*u.m
d_oap6_lyotstop = 0.687567667550736*u.m
d_lyotstop_oap7 = 0.401748843470518*u.m
fl_oap7 = 0.708251083480054*u.m
diam_oap7 = 0.06*u.m
d_oap7_fieldstop = 0.708251083480054*u.m  
d_fieldstop_oap8 = 0.210985967281651*u.m
fl_oap8 = 0.210985967281651*u.m
diam_oap8 = 0.06*u.m
d_oap8_pupil = 0.238185804200797*u.m
d_oap8_filter = 0.368452268225530*u.m
diam_filter = 0.01*u.m
d_filter_lens = 0.170799548215162*u.m
fl_lens = 0.246017378417573*u.m + 0.050001306014153*u.m
diam_lens = 0.01*u.m
d_lens_fold4 = 0.246017378417573*u.m
diam_fold4 = 0.02*u.m
d_fold4_image = 0.050001578514650*u.m
fl_pupillens = 0.149260576823040*u.m   

primary = poppy.QuadraticLens(fl_pri, name='Primary')
secondary = poppy.QuadraticLens(fl_sec, name='Secondary')
fold1 = poppy.CircularAperture(radius=diam_fold1/2,name="Fold 1")
m3 = poppy.QuadraticLens(fl_m3, name='M3')
m4 = poppy.QuadraticLens(fl_m4, name='M4')
m5 = poppy.QuadraticLens(fl_m5, name='M5')
fold2 = poppy.CircularAperture(radius=diam_fold2/2,name="Fold 2")
fsm = poppy.CircularAperture(radius=0.5*u.m,name="FSM")
oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
focm = poppy.CircularAperture(radius=0.5*u.m,name="FOCM")
oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold 3")
oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
fpm_plane = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FPM (None)') # an intermediate plane to obtain wavefront
oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
fieldstop_plane = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Fieldstop (None)')
fieldstop = poppy.CircularAperture(radius=0.5*u.m, name='Fieldstop')
oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
lens = poppy.QuadraticLens(fl_lens, name='LENS')
fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold 4")
image = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='focus')

print('\nInitializing all FITSOpticalElements from these directories...')
opticsdir = Path('/groups/douglase/webbpsf-data/CGI/optics/F730'); print('SPC730 Optics directory:',opticsdir)
opddir = Path('/groups/douglase/webbpsf-data/CGI/OPD'); print('OPD directory:',opddir,'\n')

# get the OPD data
primary_opd = poppy.FITSOpticalElement('Primary OPD',
                                       opd=str(opddir/'wfirst_phaseb_PRIMARY_phase_error_V1.0.fits'),
                                       opdunits='meters', 
                                       planetype=PlaneType.intermediate)
g2o_opd = poppy.FITSOpticalElement('G2O OPD',
                                   opd=str(opddir/'wfirst_phaseb_GROUND_TO_ORBIT_4.2X_phase_error_V1.0.fits'),
                                   opdunits='meters', 
                                   planetype=PlaneType.intermediate)
secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                         opd=str(opddir/'wfirst_phaseb_SECONDARY_phase_error_V1.0.fits'),
                                         opdunits='meters', 
                                         planetype=PlaneType.intermediate)
fold1_opd = poppy.FITSOpticalElement('Fold-1 OPD',
                                     opd=str(opddir/'wfirst_phaseb_FOLD1_phase_error_V1.0.fits'),
                                     opdunits='meters',
                                     planetype=PlaneType.intermediate)
m3_opd = poppy.FITSOpticalElement('M3 OPD',
                                  opd=str(opddir/'wfirst_phaseb_M3_phase_error_V1.0.fits'),
                                  opdunits='meters',
                                  planetype=PlaneType.intermediate)
m4_opd = poppy.FITSOpticalElement('M4 OPD',
                                  opd=str(opddir/'wfirst_phaseb_M4_phase_error_V1.0.fits'),
                                  opdunits='meters',
                                  planetype=PlaneType.intermediate)
m5_opd = poppy.FITSOpticalElement('M5 OPD',
                                  opd=str(opddir/'wfirst_phaseb_M5_phase_error_V1.0.fits'),
                                  opdunits='meters', 
                                  planetype=PlaneType.intermediate)
fold2_opd = poppy.FITSOpticalElement('Fold-2 OPD',
                                     opd=str(opddir/'wfirst_phaseb_FOLD2_phase_error_V1.0.fits'),
                                     opdunits='meters',
                                     planetype=PlaneType.intermediate)
fsm_opd = poppy.FITSOpticalElement('FSM OPD',
                                   opd=str(opddir/'wfirst_phaseb_FSM_phase_error_V1.0.fits'),
                                   opdunits='meters', 
                                   planetype=PlaneType.intermediate)
oap1_opd = poppy.FITSOpticalElement('OAP1 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP1_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
focm_opd = poppy.FITSOpticalElement('FOCM OPD',
                                    opd=str(opddir/'wfirst_phaseb_FOCM_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
oap2_opd = poppy.FITSOpticalElement('OAP2 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP2_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
dm1_opd = poppy.FITSOpticalElement('DM1 OPD',
                                   opd=str(opddir/'wfirst_phaseb_DM1_phase_error_V1.0.fits'),
                                   opdunits='meters',
                                   planetype=PlaneType.intermediate)
dm2_opd = poppy.FITSOpticalElement('DM2 OPD',
                                   opd=str(opddir/'wfirst_phaseb_DM2_phase_error_V1.0.fits'),
                                   opdunits='meters',
                                   planetype=PlaneType.intermediate)
oap3_opd = poppy.FITSOpticalElement('OAP3 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP3_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
fold3_opd = poppy.FITSOpticalElement('Fold-3 OPD',
                                     opd=str(opddir/'wfirst_phaseb_FOLD3_phase_error_V1.0.fits'),
                                     opdunits='meters',
                                     planetype=PlaneType.intermediate)
oap4_opd = poppy.FITSOpticalElement('OAP4 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP4_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
spm_opd = poppy.FITSOpticalElement('SPM OPD',
                                   opd=str(opddir/'wfirst_phaseb_PUPILMASK_phase_error_V1.0.fits'),
                                   opdunits='meters',
                                   planetype=PlaneType.intermediate)
oap5_opd = poppy.FITSOpticalElement('OAP5 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP5_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
oap6_opd = poppy.FITSOpticalElement('OAP6 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP6_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
oap7_opd = poppy.FITSOpticalElement('OAP7 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP7_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
oap8_opd = poppy.FITSOpticalElement('OAP8 OPD',
                                    opd=str(opddir/'wfirst_phaseb_OAP8_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
filter_opd = poppy.FITSOpticalElement('Filter OPD',
                                      opd=str(opddir/'wfirst_phaseb_FILTER_phase_error_V1.0.fits'),
                                      opdunits='meters',
                                      planetype=PlaneType.intermediate)
lens_opd = poppy.FITSOpticalElement('LENS OPD',
                                    opd=str(opddir/'wfirst_phaseb_LENS_phase_error_V1.0.fits'),
                                    opdunits='meters',
                                    planetype=PlaneType.intermediate)
fold4_opd = poppy.FITSOpticalElement('Fold-4 OPD',
                                     opd=str(opddir/'wfirst_phaseb_FOLD4_phase_error_V1.1.fits'),
                                     opdunits='meters', 
                                     planetype=PlaneType.intermediate)

def run_model(npix=1024,
              oversample=2,
              lambda_m=None,
              tilts=(0,0),
              use_fpm=True,
              use_errors=False,
              use_dms=False, 
              use_fieldstop=False,
              centering='ADJUSTABLE',
              display_intermediates=False):
    reload(poppy); reload(misc)
    
    if lambda_m==None:
        lambda_m = 730e-9*u.m
    D = 2.3633372*u.m

    pupil = poppy.FITSOpticalElement('Roman Pupil', str(opticsdir/'pupil_SPC-20190130_rotated.fits'), 
                                     planetype=PlaneType.pupil)

    SPM = poppy.FITSOpticalElement('Shaped Pupil Mask', str(opticsdir/'SPM_SPC-20190130.fits'),
#                                    pixelscale=1.70005966366624e-05, 
                                   planetype=PlaneType.pupil)
    
#     FPM = poppy.FITSFPMElement('BOWTIE FPM', 
#                                str(opticsdir/'fpm_0.05lamdivD.fits'), 
# #                                opd=str(opticsdir/'fpm_0.05lamdivD.fits'), opdunits='meter',
#                                wavelength_c=730e-9*u.m, 
#                                ep_diam=D, 
#                                pixelscale_lamD=0.05,
#                                centering=centering,)
    FPM = poppy.FITSFPMElement('BOWTIE FPM', 
                               str(opticsdir/'FPM_res100_SPC-20190130.fits'), 
#                                opd=str(opticsdir/'fpm_0.05lamdivD.fits'), opdunits='meter',
                               wavelength_c=730e-9*u.m, 
                               ep_diam=D, 
                               pixelscale_lamD=0.01,
                               centering=centering,)
    FPM.opd *= 2.5e-7
    
    LS = poppy.FITSOpticalElement('Lyot Stop', str(opticsdir/'LS_SPC-20190130.fits'), 
#                                   pixelscale=1.7000357988404796e-05,
                                  planetype=PlaneType.pupil)

    # Get the DM data
    dm1 = poppy.FITSOpticalElement('DM1', 
                                   opd=str(opticsdir/'spc-spec_long_with_aberrations_dm1.fits'), 
                                   opdunits='meters', 
                                   planetype=PlaneType.intermediate)
    dm2 = poppy.FITSOpticalElement('DM2',
                                   opd=str(opticsdir/'spc-spec_long_with_aberrations_dm2.fits'),
                                   opdunits='meters', 
                                   planetype=PlaneType.intermediate)
    
    clear_output()
    
    fig=plt.figure(figsize=(4,4)); pupil.display(); plt.close(); display(fig)
    fig=plt.figure(figsize=(4,4)); SPM.display(); plt.close(); display(fig)
    fig=plt.figure(figsize=(10,4)); FPM.display(what='both'); plt.close(); display(fig)
    fig=plt.figure(figsize=(4,4)); LS.display(); plt.close(); display(fig)
    fig=plt.figure(figsize=(10,4)); dm1.display(what='both'); plt.close(); display(fig)
    fig=plt.figure(figsize=(10,4)); dm2.display(what='both'); plt.close(); display(fig)
    fig=plt.figure(figsize=(10,4)); primary_opd.display(what='both'); plt.close(); display(fig)
    
    # proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
    xtilt,ytilt = tilts
    tilt = poppy.ZernikeWFE(radius=D/2, coefficients=[0, xtilt, ytilt], aperture_stop=False)
    fig=plt.figure(figsize=(10,4)); tilt.display(what='both'); plt.close(); display(fig)

    # create the optical system
    beam_ratio = 1/oversample
    spc = poppy.FresnelOpticalSystem(name='SPC730', pupil_diameter=D,
                                     npix=npix, beam_ratio=beam_ratio, verbose=True)

    spc.add_optic(pupil)
    spc.add_optic(tilt)
    
    spc.add_optic(primary)
    if use_errors: spc.add_optic(primary_opd)
    if use_errors: spc.add_optic(g2o_opd)

    spc.add_optic(secondary, distance=d_pri_sec)
    spc.add_optic(poppy.CircularAperture(radius=diam_sec/2,name="Secondary aperture"))
    if use_errors: spc.add_optic(secondary_opd)

    spc.add_optic(fold1, distance=d_sec_fold1)
    if use_errors: spc.add_optic(fold1_opd)

    spc.add_optic(m3, distance=d_fold1_m3)
    spc.add_optic(poppy.CircularAperture(radius=diam_m3/2,name="M-3 aperture"))
    if use_errors: spc.add_optic(m3_opd)

    spc.add_optic(m4, distance=d_m3_m4)
    spc.add_optic(poppy.CircularAperture(radius=diam_m4/2,name="M-4 aperture"))
    if use_errors: spc.add_optic(m4_opd)

    spc.add_optic(m5, distance=d_m4_m5)
    spc.add_optic(poppy.CircularAperture(radius=diam_m5/2,name="M-5 aperture"))
    if use_errors: spc.add_optic(m5_opd)

    spc.add_optic(fold2, distance=d_m5_fold2)
    if use_errors: spc.add_optic(fold2_opd)

    spc.add_optic(fsm, distance=d_fold2_fsm)
    if use_errors: spc.add_optic(fsm_opd)

    spc.add_optic(oap1, distance=d_fsm_oap1)
    spc.add_optic(poppy.CircularAperture(radius=diam_oap1/2,name="OAP1 aperture"))
    if use_errors: spc.add_optic(oap1_opd)

    spc.add_optic(focm, distance=d_oap1_focm)
    if use_errors: spc.add_optic(focm_opd)

    if use_dms:
        print('Use DMs')
        spc.add_optic(oap2, distance=d_focm_oap2)
        spc.add_optic(poppy.CircularAperture(radius=diam_oap2/2,name="OAP2 aperture"))
        if use_errors: spc.add_optic(oap2_opd)

        spc.add_optic(dm1, distance=d_oap2_dm1)
        if use_errors: spc.add_optic(dm1_opd)

        spc.add_optic(dm2, distance=d_dm1_dm2)
        if use_errors: spc.add_optic(dm2_opd)

        spc.add_optic(oap3, distance=d_dm2_oap3)
    else:
        print('Not using DMs.')
        spc.add_optic(oap2, distance=d_focm_oap2)
        spc.add_optic(poppy.CircularAperture(radius=diam_oap2/2,name="OAP2 aperture"))
        if use_errors: spc.add_optic(oap2_opd)

        spc.add_optic(oap3, distance=d_oap2_dm1 + d_dm1_dm2 + d_dm2_oap3)

    spc.add_optic(poppy.CircularAperture(radius=diam_oap3/2,name="OAP3 aperture"))
    if use_errors: spc.add_optic(oap3_opd)

    spc.add_optic(fold3, distance=d_oap3_fold3)
    if use_errors: spc.add_optic(fold3_opd)

    spc.add_optic(oap4, distance=d_fold3_oap4)
    spc.add_optic(poppy.CircularAperture(radius=diam_oap4/2,name="OAP4 aperture"))
    if use_errors: spc.add_optic(oap4_opd)

    spc.add_optic(SPM, distance=d_oap4_pupilmask)

    spc.add_optic(oap5, distance=d_pupilmask_oap5)
    if use_errors: spc.add_optic(oap5_opd)

    if use_fpm: spc.add_optic(FPM, distance=d_oap5_fpm)
    else: spc.add_optic(fpm_plane, distance=d_oap5_fpm)

    spc.add_optic(oap6, distance=d_fpm_oap6)
    spc.add_optic(poppy.CircularAperture(radius=diam_oap6/2,name="OAP6 aperture"))
    if use_errors: spc.add_optic(oap6_opd)

    spc.add_optic(LS, distance=d_oap6_lyotstop)

    spc.add_optic(oap7, distance=d_lyotstop_oap7)
    spc.add_optic(poppy.CircularAperture(radius=diam_oap7/2,name="OAP7 aperture"))
    if use_errors: spc.add_optic(oap7_opd)

    if use_fieldstop: spc.add_optic(fieldstop, distance=d_oap7_fieldstop)
    else: spc.add_optic(fieldstop_plane, distance=d_oap7_fieldstop)

    spc.add_optic(oap8, distance=d_fieldstop_oap8)
    spc.add_optic(poppy.CircularAperture(radius=diam_oap8/2,name="OAP8 aperture"))
    if use_errors: spc.add_optic(oap8_opd)

    spc.add_optic(filt, distance=d_oap8_filter)
    if use_errors: spc.add_optic(filter_opd)

    spc.add_optic(lens, distance=d_filter_lens)
    if use_errors: spc.add_optic(lens_opd)

    spc.add_optic(fold4, distance=d_lens_fold4)
    if use_errors: spc.add_optic(fold4_opd)

    spc.add_optic(image, distance=d_fold4_image)

    spc.describe()

    # calculate the PSF of the second optical system
    fig=plt.figure(figsize=(15,15))
    psf,wfs = spc.calc_psf(wavelength=lambda_m, 
                            display_intermediates=display_intermediates, 
                            return_intermediates=True,)
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.close(); display(fig)
    
    ''' 
    End of all propagation, just plotting and analyses.
    '''
    
    # compare to final wavefront from proper by interpolating pixelscales
    wfnum = -1
    npop = wfs[wfnum].wavefront.shape[0]
    pop_wf = wfs[wfnum].wavefront
    pop_samp = wfs[wfnum].pixelscale
    print(pop_wf.shape, pop_samp)
    
    misc.myimshow2(np.abs(pop_wf)**2, np.angle(pop_wf),
                   'PSF Intensity', 'PSF Phase',
                   n=128,
                   lognorm1=True,
                   pxscl=wfs[wfnum].pixelscale)
    
    wf_fpath = 'spc730-fresnel-wavefronts/wf_'
    if use_errors: wf_fpath += 'ab_'
    wf_fpath += 'psf_'
    if use_fpm==False: wf_fpath += 'nofpm_'
    if xtilt!=0: wf_fpath += 'offax_'
    wf_fpath += 'proper.fits'
    print(wf_fpath)
    
    proper_wf = fits.getdata(wf_fpath)
    proper_hdr = fits.getheader(wf_fpath)
    prop_samp = proper_hdr['PIXELSCL']*u.m/u.pix
    prop_samp_lamD = proper_hdr['PIXSCLLD'] # pixelscale in lam/D
    nprop = proper_wf.shape[1]
    
    # perform the interpolation using proper's prop_magnify function as this can conserve wavefront amplitude
    mag = pop_samp.value / prop_samp.value
    pop_samp = pop_samp/mag
    output_dim = nprop
    pop_wf = proper.prop_magnify( pop_wf, mag, output_dim, AMP_CONSERVE=True ).T
    print(proper_wf[0].shape, prop_samp)
    
    # get normalization values
    vmax=np.max(np.abs(pop_wf)**2)
    if np.max(proper_wf[0])>vmax:
        vmax=np.max(proper_wf[0])
    vmin=np.min(np.abs(pop_wf)**2)
    if np.min(proper_wf[0])<vmin:
        vmin=np.min(proper_wf[0])
    norm=LogNorm(vmin=vmin,vmax=vmax)
    
    innwa = 3/prop_samp_lamD*prop_samp.value
    outwa = 9/prop_samp_lamD*prop_samp.value
    
    ## first plot
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(np.abs(pop_wf)**2, np.angle(pop_wf),
                   'POPPY Final Intensity', 'POPPY Final Phase',
                   pxscl=pop_samp,
                   lognorm1=True, vmin1=vmin, vmax1=vmax,
                   patches1=patches1, patches2=patches2,
                   wspace=0.45)

    ## second plot
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(proper_wf[0], proper_wf[1],
                   'PROPER Final Intensity', 'PROPER Final Phase',
                   pxscl=prop_samp,
                   lognorm1=True, vmin1=vmin, vmax1=vmax,
                   patches1=patches1, patches2=patches2,
                   wspace=0.45)
    
    print(np.min(np.abs(pop_wf)**2), np.max(np.abs(pop_wf)**2))
    print(np.min(proper_wf[0]), np.max(proper_wf[0]))
    
    print('Total flux from POPPY: ', np.sum(np.abs(pop_wf)**2))
    print('Total flux from PROPER: ', np.sum(proper_wf[0]))
    
    ## difference plot
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(np.abs(np.abs(pop_wf)**2 - proper_wf[0]), np.abs(np.angle(pop_wf) - proper_wf[1]),
                   'Final Intensity Difference', 'Final Phase Difference',
                   pxscl=prop_samp,
                   lognorm1=True,
                   patches1=patches1, patches2=patches2,
                   wspace=0.45)
    
    return psf, wfs
    
    
    
    


