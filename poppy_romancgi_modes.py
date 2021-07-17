import poppy
from poppy.poppy_core import PlaneType
import proper
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, clear_output

import os
import time
import glob
from pathlib import Path

from importlib import reload
import misc
import polmap

# this section defines various optic focal lengths, diameters, and distances between optics.
D = 2.3633372*u.m
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

def make_inwave(mode, cgi_dir, D, lambda_c_m, lambda_m, npix, oversample, offsets, polaxis, display=True):
    ''' 
    Initialize the monochromatic input wavefront
        mode: 'HLC575', 'SPC730', or 'SPC825'
        cgi_dir: the directory to the POPPY CGI data files
        D: pupil diameter of the intended FresnelOpticalSystem and therefore the input wavefront
        lambda_c_m: central wavelength of the mode in units of m
        lambda_m: intended wavelength of the wavefront
        npix: number of pixels across the pupil diameter
        oversample: oversample factor of the wavefront
        offsets: source offsets given a tuple as such (x_offset, y_offset)
        polaxis: polarization axis to determine polarization aberrations with the polmap function
        display: display the input wavefront or not
    ''' 
    wfin = poppy.FresnelWavefront(beam_radius=D/2, wavelength=lambda_m, npix=npix, oversample=oversample)
    
    if polaxis!=0: 
        print('Employing polarization aberrations.')
        polfile = cgi_dir/'optics'/'pol'/'new_toma' # the file extension is added by the polmap function call
        if mode=='HLC575': polmap.polmap( wfin, polfile, 309, polaxis )
        else: polmap.polmap( wfin, polfile, npix, polaxis )
    else:
        print('Not employing polarization aberrations.')
    
    xoffset = offsets[0]
    yoffset = offsets[1]
    xoffset_lam = -xoffset * (lambda_c_m / lambda_m).value * (D/2.3633372*u.m).value # maybe use negative sign
    yoffset_lam = -yoffset * (lambda_c_m / lambda_m).value * (D/2.3633372*u.m).value
    n = npix*oversample
    x = np.tile( (np.arange(n)-n//2)/(npix/2.0), (n,1) )
    y = np.transpose(x)
    wfin.wavefront = wfin.wavefront * np.exp(complex(0,1) * np.pi * (xoffset_lam * x + yoffset_lam * y))
    
    if display:
        misc.myimshow2(np.abs(wfin.wavefront)**2, np.angle(wfin.wavefront),
                       'Input Wave Intensity', 'Input Wave Phase',
                       pxscl=wfin.pixelscale, 
                       cmap1='gist_heat', cmap2='viridis')
    
    return wfin

def resample_psf(pop_psf, pixelscale, npix): 
    ''' 
    Resample a POPPY PSF by providing the following:
        pop_psf: the FresnelWavefront object of the PSF
        pixelscale: the desired output pixelscale in units of m/pix
        npix: the desired output dimension of the PSF
    '''
    pop_psf_wf = pop_psf.wavefront
    pop_pxscl = pop_psf.pixelscale.value
    print('Input POPPY wavefront pixelscale: ', pop_pxscl*u.m/u.pix)
    mag = pop_pxscl/pixelscale
    resamp_pop_psf_wf = proper.prop_magnify(pop_psf_wf, mag, npix, AMP_CONSERVE=True)
    print('Interpolated POPPY wavefront pixelscale: ', pop_pxscl/mag*u.m/u.pix)
    return resamp_pop_psf_wf, mag

def compare_psfs(pop_psf, prop_psf_fpath, vmin=None, vmax=None):
    '''
    Useful function to directly display and compare a POPPY PSF and a PROPER PSF the POPPY PSF will automatically be rotated 
    by 180degrees to match the sign convention of PROPER.
        pop_psf: the FresnelWavefront object of the PSF
        prop_psf_fpath: the file path to the desired PROPER PSF which is saved as a FITS file in the proper-psfs directory 
        vmin and vmax: set the minimum and maximum values for the colorbar
    '''
    reload(misc)
    prop_psf_fname = str(prop_psf_fpath)
    if 'hlc' in prop_psf_fname:
        psf_title = 'HLC, '
        iwa=3; owa=9
    if 'spc-spec' in prop_psf_fname:
        psf_title = 'SPC-Spec, '
        iwa=3; owa=9
    elif 'spc-wide' in prop_psf_fname: 
        psf_title = 'SPC-Wide, '
        iwa=5.4; owa=20
    psf_title += str(int(pop_psf.wavelength.value*1e9))+'nm,'
    if '_onax' in prop_psf_fname: psf_title += ' on-axis,\n'
    else: psf_title += ' off-axis,\n'
    if '_nofs' in prop_psf_fname: psf_title += ' no FS,'
    if '_dms' in prop_psf_fname: psf_title += ' with DMs,'
    else: psf_title += ' without DMs,'
    if '_opds' in prop_psf_fname: psf_title += ' with OPDs,'
    else: psf_title += ' without OPDs,'
    prop_wf = fits.getdata(Path(prop_psf_fname))
    prop_int = prop_wf[0] # the initial PROPER PSF wavefront array is 3D, first entry of first dimention is the intensity.
    prop_phs = prop_wf[1] # second entry of first dimension is the phase
    prop_pxscl = fits.getheader(prop_psf_fname)['PIXELSCL'] # saved PROPER wfs have pixelscales saved in both m/pix and lambda/D/pix
    prop_pxscl_lamD = fits.getheader(prop_psf_fname)['PIXSCLLD']
    print('PROPER wavefront pixelscale: ', prop_pxscl*u.m/u.pix)
    print('PROPER wavefront pixelscale in \u03BB/D: ', prop_pxscl_lamD)
    
    pop_psf_wf, mag = resample_psf(pop_psf, prop_pxscl, prop_int.shape[0])
    
    # rotate poppy psf 180degree since there is a sign convention difference between POPPY and PROPER
    pop_psf_wf_r = scipy.ndimage.rotate(np.real(pop_psf_wf), 180)
    pop_psf_wf_i = scipy.ndimage.rotate(np.imag(pop_psf_wf), 180)
    pop_psf_wf = pop_psf_wf_r + 1j*pop_psf_wf_i
    pop_int = np.abs(pop_psf_wf)**2
    pop_phs = np.angle(pop_psf_wf)
    
    if vmin is None:
        vmin = np.min(pop_int)
        if np.min(prop_int) < vmin: vmin = np.min(prop_int)
    if vmax is None:
        vmax = np.max(pop_int)
        if np.max(prop_int) > vmax: vmax = np.max(prop_int)
    
    innwa = iwa/prop_pxscl_lamD*prop_pxscl*1000 # inner and outer working angles in units of m
    outwa = owa/prop_pxscl_lamD*prop_pxscl*1000
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(pop_int, prop_int, 'POPPY PSF: '+psf_title, 'PROPER PSF: '+psf_title,
                   pxscl=(prop_pxscl*u.m/u.pix).to(u.mm/u.pix),
                   cmap1='gist_heat', cmap2='gist_heat',
                   lognorm1=True, lognorm2=True, vmin1=vmin, vmax1=vmax, vmin2=vmin, vmax2=vmax,
                   patches1=patches1, patches2=patches2,)
    
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(pop_phs, prop_phs, 'POPPY PSF Phase', 'PROPER PSF Phase',
                   pxscl=(prop_pxscl*u.m/u.pix).to(u.mm/u.pix),
                   cmap1='viridis', cmap2='viridis',
                   patches1=patches1, patches2=patches2,)
    
    # difference plots
    int_diff = np.abs(pop_int - prop_int)
    phs_diff = pop_phs-prop_phs
    patches1 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    patches2 = [Circle((0,0),innwa,edgecolor='c', facecolor='none',lw=1),
                Circle((0,0),outwa,edgecolor='c', facecolor='none',lw=1)]
    misc.myimshow2(int_diff, phs_diff, 'PSF Intesnity Difference', 'PSF Phase Difference',
                   pxscl=(prop_pxscl*u.m/u.pix).to(u.mm/u.pix),
                   cmap1='gist_heat', cmap2='viridis',
                   lognorm1=True, vmax1=vmax,
                   patches1=patches1, patches2=patches2)
     
def run_model(mode='HLC575',
              oversample=2,
              lambda_m=None,
              offsets=(0,0),
              use_fpm=True,
              use_opds=False,
              use_dms=False, 
              use_fieldstop=False,
              use_apertures=False,
              polaxis=0,
              cgi_dir=None,
              display_mode=False,
              display_inwave=False,
              display_intermediates=False,
              display_fpm=False,
              display_psf=True):
    '''
    This is the function used to calculate a PSF. WebbPSF would make this easier as you can initialize a mode, alter options, and then 
    calculate a PSF rather than using one function to initialize a mode and putting in options as kwargs to calculate the desired PSF.
        mode: which mode to calculate a PSF for; 'HLC575', 'SPC730', or 'SPC825'
        oversample: how much oversampling for the FresnelOpticalSystem's pupil diameter to use, default is 2 for all modes
        lambda_m: wavelength to propagate; if None, the wavelength used will be the modes central wavelength
        offsets: source offset coordinates provided as a tuple as such (x_offset, y_offset)
        use_fpm: use the modes FPM or not, default is True
        use_opds: use individual optic OPDs, default is False
        use_dms: use DM maps for the given mode, default is False; if True, the DM maps for the correct scenario are applied. The SPC modes
                 each have one set of DM maps, for when polaxis is 10 and all OPDs are used. The HLC mode has two sets of DM maps, one for 
                 when no polaxis or OPDs are used since the HLC uses the DMs ton make a dark-hole anyway and another set for when polaxis is 
                 10 and when OPDs are used. 
        use_fieldstop: use the fieldstop or not, only useable for the HLC mode
        use_apertures: use individual optic apertures
        polaxis: what polarization axis to detremine the polarization aberrations for; will use the polmap function to do so; default is 0, 
                 meaning no polarization aberration are used
        cgi_dir: path to the directory containing the POPPY CGI data files
        display_mode: display some of the mode specific optics; uses the POPPY display functionality for optical elements
        display_inwave: display the intensity and phase of the input wavefront
        display_intermediates: display all intermediate optic planes using POPPY's default display_intermediates feature
        display_fpm: display the FPM wavefront intensity after the FPM is applied or not
        display_psf: display the final PSF intensity
    '''
    reload(misc)
    reload(polmap)
    
    #################### Initialize directories and file names for the masks and OPDs
    if mode=='HLC575':
        npix = 1024
        D = 2.3633372*u.m*npix/309
        opticsdir = cgi_dir/'optics'/'F575'
        opddir = cgi_dir/'OPD-hlc575'
        pupil_fname = str(opticsdir/'run461_pupil_rotated.fits')
        
        lambda_c_m = 575e-9*u.m
        if lambda_m is None:
            lambda_m = 575e-9*u.m
        
        # this is a cumbersome way of choosing the correct complex occulter files depending on the wavelength being propagated
        fpm_lams = [5.4625e-07, 5.49444444444e-07, 5.52638888889e-07, 5.534375e-07, 5.55833333333e-07, 
                    5.59027777778e-07, 5.60625e-07, 5.62222222222e-07, 5.65416666667e-07, 5.678125e-07, 
                    5.68611111111e-07, 5.71805555556e-07, 5.75e-07, 5.78194444444e-07, 5.81388888889e-07, 
                    5.821875e-07, 5.84583333333e-07, 5.87777777778e-07, 5.89375e-07, 5.90972222222e-07,
                    5.94166666667e-07, 5.965625e-07, 5.97361111111e-07, 6.00555555556e-07, 6.0375e-07 ]
        fpm_lams_strs = ['5.4625e-07', '5.49444444444e-07', '5.52638888889e-07', '5.534375e-07', '5.55833333333e-07', 
                         '5.59027777778e-07', '5.60625e-07', '5.62222222222e-07', '5.65416666667e-07', '5.678125e-07', 
                         '5.68611111111e-07', '5.71805555556e-07', '5.75e-07', '5.78194444444e-07', '5.81388888889e-07', 
                         '5.821875e-07', '5.84583333333e-07', '5.87777777778e-07', '5.89375e-07', '5.90972222222e-07', 
                         '5.94166666667e-07', '5.965625e-07', '5.97361111111e-07', '6.00555555556e-07', '6.0375e-07' ]
        lam_ind = (np.abs(lambda_m.value - np.array(fpm_lams))).argmin()
        
        fpm_name = 'Complex FPM'
        fpm_real_fname = str(opticsdir / ('run461_occ_lam' + fpm_lams_strs[lam_ind] + 'theta6.69polp_real_rotated.fits'))
        fpm_imag_fname = str(opticsdir / ('run461_occ_lam' + fpm_lams_strs[lam_ind] + 'theta6.69polp_imag_rotated.fits'))
        
        lyotstop_fname = str(opticsdir/'run461_lyot.fits')
    elif mode=='SPC730':
        npix = 1000
        D = 2.3633372*u.m
        opticsdir = cgi_dir/'optics'/'F730'
        opddir = cgi_dir/'OPD-spc730'
        pupil_fname = str(opticsdir/'pupil_SPC-20190130_rotated.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20190130.fits')
        fpm_pxscl_lamD = 0.05
        fpm_name = 'BOWTIE FPM'
        fpm_fname = str(opticsdir/'fpm_0.05lamdivD.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20190130.fits')
        
        lambda_c_m = 730e-9*u.m
        if lambda_m is None:
            lambda_m = 730e-9*u.m
    elif mode=='SPC825':
        npix = 1000
        D = 2.3633372*u.m
        opticsdir = cgi_dir/'optics'/'F825'
        opddir = cgi_dir/'OPD-spc825'
        pupil_fname = str(opticsdir/'pupil_SPC-20181220_1k_rotated.fits')
        spm_fname = str(opticsdir/'SPM_SPC-20181220_1000_rounded9_gray.fits')
        fpm_pxscl_lamD = 0.05
        fpm_name = 'ANNULAR FPM'
        fpm_fname = str(opticsdir/'fpm_0.05lamdivD.fits')
        lyotstop_fname = str(opticsdir/'LS_SPC-20181220_1k.fits')
        
        lambda_c_m = 825e-9*u.m
        if lambda_m is None:
            lambda_m = 825e-9*u.m
        
    #################### Initialize mode specific optics 
    pupil = poppy.FITSOpticalElement('Roman Pupil', pupil_fname, planetype=PlaneType.pupil)
    LS = poppy.FITSOpticalElement('Lyot Stop', lyotstop_fname, planetype=PlaneType.pupil)
    if mode=='SPC730' or mode=='SPC825': SPM = poppy.FITSOpticalElement('Shaped Pupil Mask', spm_fname, planetype=PlaneType.pupil)
    elif mode=='HLC575': SPM = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='SPM Plane (No Optic)')
      
    if use_fpm: 
        if mode=='SPC730' or mode=='SPC825':
            FPM = poppy.FITSFPMElement(fpm_name, fpm_fname, 
                                       wavelength_c=lambda_c_m, ep_diam=D, pixelscale_lamD=fpm_pxscl_lamD, centering='ADJUSTABLE',)
        elif mode=='HLC575': 
            FPM = poppy.FITSOpticalElement('Complex FPM', transmission=fpm_real_fname, planetype=PlaneType.intermediate)
            # set the actual FPM data manually with the separate occulter files for real and imaginary components
            real = fits.getdata(fpm_real_fname)
            imag = fits.getdata(fpm_imag_fname)
            fpm_phasor = real + 1j*imag
            FPM.amplitude = np.abs(fpm_phasor)
            FPM.opd = np.angle(fpm_phasor)/ (2*np.pi/fpm_lams[lam_ind])
            FPM.interp_order = 0 # set the interpolation order to 0, POPPY default is 3
    else: 
        FPM = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FPM Plane (No Optic)')
        
    # choose the correct pair of DM maps to use based on the mode and OPDs option provided
    if use_dms: 
        if mode=='HLC575':
            if use_opds:
                dm1_fname = str(opticsdir/'hlc575_dm1_map_for_opds.fits')
                dm2_fname = str(opticsdir/'hlc575_dm2_map_for_opds.fits')
                dm1 = poppy.FITSOpticalElement('DM1', opd=dm1_fname, opdunits='meters', planetype=PlaneType.intermediate)
                dm2 = poppy.FITSOpticalElement('DM2', opd=dm2_fname, opdunits='meters', planetype=PlaneType.intermediate)
            else:
                dm1_fname = str(opticsdir/'run461_dm1wfe.fits')
                dm2_fname = str(opticsdir/'run461_dm2wfe.fits')
                dm2_mask_fname = str(opticsdir/'run461_dm2mask.fits')
                dm1 = poppy.FITSOpticalElement('DM1', opd=dm1_fname, opdunits='meters', planetype=PlaneType.intermediate)
                dm2 = poppy.FITSOpticalElement('DM2', opd=dm2_fname, opdunits='meters', planetype=PlaneType.intermediate)
        elif mode=='SPC730':
            dm1_fname = str(opticsdir/'spc730_dm1_map_for_opds.fits')
            dm2_fname = str(opticsdir/'spc730_dm2_map_for_opds.fits')
            dm1 = poppy.FITSOpticalElement('DM1', opd=dm1_fname, opdunits='meters', planetype=PlaneType.intermediate)
            dm2 = poppy.FITSOpticalElement('DM2', opd=dm2_fname, opdunits='meters', planetype=PlaneType.intermediate)
        elif mode=='SPC825':
            dm1_fname = str(opticsdir/'spc825_dm1_map_for_opds.fits')
            dm2_fname = str(opticsdir/'spc825_dm2_map_for_opds.fits')
            dm1 = poppy.FITSOpticalElement('DM1', opd=dm1_fname, opdunits='meters', planetype=PlaneType.intermediate)
            dm2 = poppy.FITSOpticalElement('DM2', opd=dm2_fname, opdunits='meters', planetype=PlaneType.intermediate)
    else: 
        dm1 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM1 Plane (No Optic)')
        dm2 = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='DM2 Plane (No Optic)')
    
    if mode=='HLC575' and use_fieldstop:
        radius = 9 / (309/(npix*oversample)) * (lambda_c_m/lambda_m) * 3.61431587167163e-06*u.m
        fieldstop = poppy.CircularAperture(radius=radius, name='Fieldstop')
    else: 
        fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Fieldstop Plane (No Optic)')
    
    #################### Define optics
    primary = poppy.QuadraticLens(fl_pri, name='Primary')
    secondary = poppy.QuadraticLens(fl_sec, name='Secondary')
    fold1 = poppy.CircularAperture(radius=diam_fold1/2,name="Fold 1")
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m4, name='M4')
    m5 = poppy.QuadraticLens(fl_m5, name='M5')
    fold2 = poppy.CircularAperture(radius=diam_fold2/2,name="Fold 2")
    fsm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FSM')
    oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
    focm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FOCM')
    oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
    oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
    fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold 3")
    oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
    oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
    oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
    oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
    oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
    filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
    lens = poppy.QuadraticLens(fl_lens, name='LENS')
    fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold 4")
    image = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='focus')
    
    #################### Initialize OPDs if using them in the FresnelOpticalSystem
    if use_opds:
        primary_opd = poppy.FITSOpticalElement('Primary OPD',
                                               opd=str(opddir/'wfirst_phaseb_PRIMARY_phase_error_V1.0.fits'), opdunits='meters', 
                                               planetype=PlaneType.intermediate)
        g2o_opd = poppy.FITSOpticalElement('G2O OPD',
                                           opd=str(opddir/'wfirst_phaseb_GROUND_TO_ORBIT_4.2X_phase_error_V1.0.fits'), opdunits='meters', 
                                           planetype=PlaneType.intermediate)
        secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                                 opd=str(opddir/'wfirst_phaseb_SECONDARY_phase_error_V1.0.fits'), opdunits='meters', 
                                                 planetype=PlaneType.intermediate)
        fold1_opd = poppy.FITSOpticalElement('Fold-1 OPD',
                                             opd=str(opddir/'wfirst_phaseb_FOLD1_phase_error_V1.0.fits'), opdunits='meters',
                                             planetype=PlaneType.intermediate)
        m3_opd = poppy.FITSOpticalElement('M3 OPD',
                                          opd=str(opddir/'wfirst_phaseb_M3_phase_error_V1.0.fits'), opdunits='meters',
                                          planetype=PlaneType.intermediate)
        m4_opd = poppy.FITSOpticalElement('M4 OPD',
                                          opd=str(opddir/'wfirst_phaseb_M4_phase_error_V1.0.fits'), opdunits='meters',
                                          planetype=PlaneType.intermediate)
        m5_opd = poppy.FITSOpticalElement('M5 OPD',
                                          opd=str(opddir/'wfirst_phaseb_M5_phase_error_V1.0.fits'), opdunits='meters', 
                                          planetype=PlaneType.intermediate)
        fold2_opd = poppy.FITSOpticalElement('Fold-2 OPD',
                                             opd=str(opddir/'wfirst_phaseb_FOLD2_phase_error_V1.0.fits'), opdunits='meters',
                                             planetype=PlaneType.intermediate)
        fsm_opd = poppy.FITSOpticalElement('FSM OPD',
                                           opd=str(opddir/'wfirst_phaseb_FSM_phase_error_V1.0.fits'), opdunits='meters', 
                                           planetype=PlaneType.intermediate)
        oap1_opd = poppy.FITSOpticalElement('OAP1 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP1_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        focm_opd = poppy.FITSOpticalElement('FOCM OPD',
                                            opd=str(opddir/'wfirst_phaseb_FOCM_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap2_opd = poppy.FITSOpticalElement('OAP2 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP2_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        dm1_opd = poppy.FITSOpticalElement('DM1 OPD',
                                           opd=str(opddir/'wfirst_phaseb_DM1_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        dm2_opd = poppy.FITSOpticalElement('DM2 OPD',
                                           opd=str(opddir/'wfirst_phaseb_DM2_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        oap3_opd = poppy.FITSOpticalElement('OAP3 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP3_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        fold3_opd = poppy.FITSOpticalElement('Fold-3 OPD',
                                             opd=str(opddir/'wfirst_phaseb_FOLD3_phase_error_V1.0.fits'), opdunits='meters',
                                             planetype=PlaneType.intermediate)
        oap4_opd = poppy.FITSOpticalElement('OAP4 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP4_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        spm_opd = poppy.FITSOpticalElement('SPM OPD',
                                           opd=str(opddir/'wfirst_phaseb_PUPILMASK_phase_error_V1.0.fits'), opdunits='meters',
                                           planetype=PlaneType.intermediate)
        oap5_opd = poppy.FITSOpticalElement('OAP5 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP5_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap6_opd = poppy.FITSOpticalElement('OAP6 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP6_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap7_opd = poppy.FITSOpticalElement('OAP7 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP7_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        oap8_opd = poppy.FITSOpticalElement('OAP8 OPD',
                                            opd=str(opddir/'wfirst_phaseb_OAP8_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        filter_opd = poppy.FITSOpticalElement('Filter OPD',
                                              opd=str(opddir/'wfirst_phaseb_FILTER_phase_error_V1.0.fits'), opdunits='meters',
                                              planetype=PlaneType.intermediate)
        lens_opd = poppy.FITSOpticalElement('LENS OPD',
                                            opd=str(opddir/'wfirst_phaseb_LENS_phase_error_V1.0.fits'), opdunits='meters',
                                            planetype=PlaneType.intermediate)
        fold4_opd = poppy.FITSOpticalElement('Fold-4 OPD',
                                             opd=str(opddir/'wfirst_phaseb_FOLD4_phase_error_V1.1.fits'), opdunits='meters', 
                                             planetype=PlaneType.intermediate)
    clear_output()
    
    if display_mode:
        fig=plt.figure(figsize=(4,4)); pupil.display(); plt.close(); display(fig)
        fig=plt.figure(figsize=(4,4)); SPM.display(); plt.close(); display(fig)
        fig=plt.figure(figsize=(4,4)); FPM.display(); plt.close(); display(fig)
        fig=plt.figure(figsize=(4,4)); LS.display(); plt.close(); display(fig)
        fig=plt.figure(figsize=(10,4)); dm1.display(what='both'); plt.close(); display(fig)
        fig=plt.figure(figsize=(10,4)); dm2.display(what='both'); plt.close(); display(fig)
        fig=plt.figure(figsize=(4,4)); fieldstop.display(); plt.close(); display(fig)

    #################### Create the optical system
    beam_ratio = 1/oversample
    fosys = poppy.FresnelOpticalSystem(name=mode, pupil_diameter=D, npix=npix, beam_ratio=beam_ratio, verbose=True)

    fosys.add_optic(pupil)

    fosys.add_optic(primary)
    if use_opds: 
        fosys.add_optic(primary_opd)
        fosys.add_optic(g2o_opd)

    fosys.add_optic(secondary, distance=d_pri_sec)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_sec/2,name="Secondary aperture"))
    if use_opds: fosys.add_optic(secondary_opd)

    fosys.add_optic(fold1, distance=d_sec_fold1)
    if use_opds: fosys.add_optic(fold1_opd)

    fosys.add_optic(m3, distance=d_fold1_m3)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_m3/2,name="M-3 aperture"))
    if use_opds: fosys.add_optic(m3_opd)

    fosys.add_optic(m4, distance=d_m3_m4)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_m4/2,name="M-4 aperture"))
    if use_opds: fosys.add_optic(m4_opd)

    fosys.add_optic(m5, distance=d_m4_m5)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_m5/2,name="M-5 aperture"))
    if use_opds: fosys.add_optic(m5_opd)

    fosys.add_optic(fold2, distance=d_m5_fold2)
    if use_opds: fosys.add_optic(fold2_opd)

    fosys.add_optic(fsm, distance=d_fold2_fsm)
    if use_opds: fosys.add_optic(fsm_opd)

    fosys.add_optic(oap1, distance=d_fsm_oap1)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap1/2,name="OAP1 aperture"))
    if use_opds: fosys.add_optic(oap1_opd)

    fosys.add_optic(focm, distance=d_oap1_focm)
    if use_opds: fosys.add_optic(focm_opd) 

    fosys.add_optic(oap2, distance=d_focm_oap2)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap2/2,name="OAP2 aperture"))
    if use_opds: fosys.add_optic(oap2_opd)

    fosys.add_optic(dm1, distance=d_oap2_dm1)
    if use_opds: fosys.add_optic(dm1_opd)

    fosys.add_optic(dm2, distance=d_dm1_dm2)
    if use_opds: fosys.add_optic(dm2_opd) 

    fosys.add_optic(oap3, distance=d_dm2_oap3)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap3/2,name="OAP3 aperture"))
    if use_opds: fosys.add_optic(oap3_opd)

    fosys.add_optic(fold3, distance=d_oap3_fold3)
    if use_opds: fosys.add_optic(fold3_opd) 

    fosys.add_optic(oap4, distance=d_fold3_oap4)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap4/2,name="OAP4 aperture"))
    if use_opds: fosys.add_optic(oap4_opd) 

    fosys.add_optic(SPM, distance=d_oap4_pupilmask)
    if use_opds: fosys.add_optic(spm_opd) 

    fosys.add_optic(oap5, distance=d_pupilmask_oap5)
    if use_opds: fosys.add_optic(oap5_opd) 

    fosys.add_optic(FPM, distance=d_oap5_fpm)

    fosys.add_optic(oap6, distance=d_fpm_oap6)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap6/2,name="OAP6 aperture"))
    if use_opds: fosys.add_optic(oap6_opd)

    fosys.add_optic(LS, distance=d_oap6_lyotstop)

    fosys.add_optic(oap7, distance=d_lyotstop_oap7)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap7/2,name="OAP7 aperture"))
    if use_opds: fosys.add_optic(oap7_opd)

    fosys.add_optic(fieldstop, distance=d_oap7_fieldstop)

    fosys.add_optic(oap8, distance=d_fieldstop_oap8)
    if use_apertures: fosys.add_optic(poppy.CircularAperture(radius=diam_oap8/2,name="OAP8 aperture"))
    if use_opds: fosys.add_optic(oap8_opd)

    fosys.add_optic(filt, distance=d_oap8_filter)
    if use_opds: fosys.add_optic(filter_opd)

    fosys.add_optic(lens, distance=d_filter_lens)
    if use_opds: fosys.add_optic(lens_opd)

    fosys.add_optic(fold4, distance=d_lens_fold4)
    if use_opds: fosys.add_optic(fold4_opd)

    fosys.add_optic(image, distance=d_fold4_image)

    #################### Calculate the PSF of the FresnelOpticalSystem
    start = time.time()
    wfin = make_inwave(mode, cgi_dir, D, lambda_c_m, lambda_m, npix, oversample, offsets, polaxis, display_inwave) # define the inwave
    psf,wfs = fosys.calc_psf(wavelength=lambda_m, 
                             display_intermediates=display_intermediates, 
                             return_final=True, return_intermediates=True,
                             inwave=wfin)
    print('PSF calculated in {:.2f}s'.format(time.time()-start))
    
    #################### Display options
    if display_fpm:
        fpmnum = 19
        if use_apertures: fpmnum = 27
        if use_opds: fpmnum = 38
        print('FPM pixelscale: ', wfs[fpmnum].pixelscale)
        misc.myimshow2(np.abs(wfs[fpmnum].wavefront)**2, np.angle(wfs[fpmnum].wavefront),
                       'Off-axis FPM Intensity', 'Off-axis FPM Phase',
                       npix=64, cmap1='gist_heat', cmap2='viridis', lognorm1=True,
                       pxscl=wfs[fpmnum].pixelscale.to(u.mm/u.pix))
        
    if display_psf:
        psfnum = -1
        print('PSF pixelscale: ', wfs[psfnum].pixelscale)
        misc.myimshow2(np.abs(wfs[psfnum].wavefront)**2, np.angle(wfs[psfnum].wavefront),
                       'PSF Intensity', 'PSF Phase',
                       npix=256, cmap1='gist_heat', cmap2='viridis',lognorm1=True,
                       pxscl=wfs[psfnum].pixelscale.to(u.mm/u.pix))

    return psf, wfs
    
    
    
    


