import numpy as np
import proper 
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.io.fits as fits
import astropy.units as u

def myimshow(arr, title=None, 
             npix=None,
             lognorm=False, vmin=None, vmax=None,
             cmap='magma',
             pxscl=None,
             patches=None,
             figsize=(4,4), dpi=125, display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr = pad_or_crop(arr, npix)
    
    if pxscl != None:
        if pxscl.unit==(u.meter/u.pix):
            vext = pxscl.value * arr.shape[0]/2
            hext = pxscl.value * arr.shape[1]/2
            extent = [-vext,vext,-hext,hext]
            ax.set_xlabel('meters')
        elif pxscl.unit==(u.mm/u.pix):
            vext = pxscl.value * arr.shape[0]/2
            hext = pxscl.value * arr.shape[1]/2
            extent = [-vext,vext,-hext,hext]
            ax.set_xlabel('millimeters')
        elif pxscl.unit==(u.arcsec/u.pix):
            vext = pxscl.value * arr.shape[0]/2
            hext = pxscl.value * arr.shape[1]/2
            extent = [-vext,vext,-hext,hext]
            ax.set_xlabel('arcsec')
    else:
        extent=None
    
    if lognorm:
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = Normalize(vmin=vmin,vmax=vmax)
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
            
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    
def myimshow2(arr1, arr2, 
              title1=None, title2=None,
              npix=None, npix1=None, npix2=None,
              pxscl=None, pxscl1=None, pxscl2=None,
              cmap1='magma', cmap2='magma',
              lognorm1=False, lognorm2=False,
              vmin1=None, vmax1=None, vmin2=None, vmax2=None, 
              patches1=None, patches2=None,
              display_fig=True, 
              return_fig=False, 
              figsize=(10,4), dpi=125, wspace=0.2):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr1 = pad_or_crop(arr1, npix)
        arr2 = pad_or_crop(arr2, npix)
    if npix1 is not None:
        arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None:
        arr2 = pad_or_crop(arr2, npix2)
    
    if pxscl is not None and pxscl!=np.nan and pxscl!=np.inf:
        vext = pxscl.value * arr1.shape[0]/2
        hext = pxscl.value * arr1.shape[1]/2
        extent1 = [-vext,vext,-hext,hext]
        
        vext = pxscl.value * arr2.shape[0]/2
        hext = pxscl.value * arr2.shape[1]/2
        extent2 = [-vext,vext,-hext,hext]
        if pxscl.unit==(u.meter/u.pix):
            ax[0].set_xlabel('meters')
            ax[1].set_xlabel('meters')
        elif pxscl.unit==(u.millimeter/u.pix):
            ax[0].set_xlabel('millimeters')
            ax[1].set_xlabel('millimeters')
        elif pxscl.unit==(u.arcsec/u.pix):
            ax[0].set_xlabel('arcsec')
            ax[1].set_xlabel('arcsec')
    else:
        if pxscl1 is not None: 
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            if pxscl1.unit==(u.meter/u.pix): ax[0].set_xlabel('meters')
            elif pxscl1.unit==(u.meter/u.pix): ax[0].set_xlabel('millimeters')
            elif pxscl1.unit==(u.arcsec/u.pix): ax[0].set_xlabel('arcsec')
        else:
            extent1=None
        if pxscl2 is not None: 
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            if pxscl2.unit==(u.meter/u.pix): ax[1].set_xlabel('meters')
            elif pxscl1.unit==(u.meter/u.pix): ax[0].set_xlabel('millimeters')
            elif pxscl2.unit==(u.arcsec/u.pix): ax[1].set_xlabel('arcsec')
        else:
            extent2=None
    
    if lognorm1: norm1 = LogNorm(vmin=vmin1,vmax=vmax1)
    else: norm1 = Normalize(vmin=vmin1,vmax=vmax1)   
    if lognorm2: norm2 = LogNorm(vmin=vmin2,vmax=vmax2)
    else: norm2 = Normalize(vmin=vmin2,vmax=vmax2)
        
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(wspace=wspace)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]

    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = np.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in

    return arr_out
        
# convenient function for saving all wavefront data at each optic of the system once a PSF is calculated
def save_waves(wfs, use_apertures, use_opds, npix=1000, wfdir=None):
    if use_apertures==False and use_opds==False:
        optics = ['pupil', 'primary', 'secondary', 'fold1', 'm3', 'm4', 'm5', 'fold2', 'fsm', 'oap1', 
                  'focm', 'oap2', 'dm1', 'dm2', 'oap3', 'fold3', 'oap4', 'spm', 'oap5', 'fpm', 'oap6',
                  'lyotstop', 'oap7', 'fieldstop', 'oap8', 'filter', 'lens', 'fold4', 'image']
        print('Saving wavefronts: ')
        for i,wf in enumerate(wfs):
            wavefront = pad_or_crop(wf.wavefront, npix)

            wf_data = np.zeros(shape=(2,npix,npix))
            wf_data[0,:,:] = np.abs(wavefront)**2
            wf_data[1,:,:] = np.angle(wavefront)

            wf_fpath = wfdir/('wf_' + optics[i] + '_poppy' + '.fits')
            hdr = fits.Header()
            hdr['PIXELSCL'] = wf.pixelscale.value

            wf_hdu = fits.PrimaryHDU(wf_data, header=hdr)
            wf_hdu.writeto(wf_fpath, overwrite=True)
            print(i, 'Saved '+optics[i]+' wavefront to ' + str(wf_fpath))
    elif use_apertures==False and use_opds==True:
        optics = ['pupil', 
                  'primary', 'primary_opd', 'g2o_opd', 
                  'secondary', 'secondary_opd',
                  'fold1', 'fold1_opd',
                  'm3', 'm3_opd',
                  'm4', 'm4_opd',
                  'm5', 'm5_opd',
                  'fold2', 'fold2_opd',
                  'fsm', 'fsm_opd',
                  'oap1', 'oap1_opd',
                  'focm', 'focm_opd',
                  'oap2', 'oap2_opd',
                  'dm1', 'dm1_opd',
                  'dm2', 'dm2_opd',
                  'oap3', 'oap3_opd',
                  'fold3', 'fold3_opd',
                  'oap4', 'oap4_opd',
                  'spm', 'spm_opd',
                  'oap5', 'oap5_opd',
                  'fpm', 
                  'oap6', 'oap6_opd',
                  'lyotstop', 
                  'oap7', 'oap7_opd',
                  'fieldstop', 
                  'oap8', 'oap8_opd',
                  'filter', 'filter_opd',
                  'lens', 'lens_opd',
                  'fold4', 'fold4_opd', 
                  'image']
        print('Saving wavefronts: ')
        for i,wf in enumerate(wfs):
            wavefront = pad_or_crop(wf.wavefront, npix)

            wf_data = np.zeros(shape=(2,npix,npix))
            wf_data[0,:,:] = np.abs(wavefront)**2
            wf_data[1,:,:] = np.angle(wavefront)

            wf_fpath = Path(wfdir)/('wf_' + optics[i] + '_poppy' + '.fits')
            hdr = fits.Header()
            hdr['PIXELSCL'] = wf.pixelscale.value

            wf_hdu = fits.PrimaryHDU(wf_data, header=hdr)
            wf_hdu.writeto(wf_fpath, overwrite=True)
            print(i, 'Saved '+optics[i]+' wavefront to ' + str(wf_fpath))
            
    print('All wavefronts saved.')
        
        
        
        