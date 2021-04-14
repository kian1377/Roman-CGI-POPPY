import numpy as np
import proper 
import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display, clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u

def myimshow(arr, title=None, 
             n=None,
             lognorm=False, vmin=None, vmax=None,
             pxscl=None,
             patches=None,
             figsize=(4,4), dpi=125, display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if n is not None:
        arr = trim(arr, n)
    
    if pxscl != None:
        vext = pxscl.value * arr.shape[0]/2
        hext = pxscl.value * arr.shape[1]/2
        extent = [-vext,vext,-hext,hext]
        
        if pxscl.unit.is_equivalent(u.meter/u.pix):
            ax.set_xlabel('meters')
        elif pxscl.unit.is_equivalent(u.arcsec/u.pix):
            ax.set_xlabel('arcsec')
        else: 
            ax.set_xlabel('lam/D')
    else:
        extent=None
    
    if lognorm:
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = Normalize(vmin=vmin,vmax=vmax)
    im = ax.imshow(arr, cmap='magma', norm=norm, extent=extent)
    
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
              n=None,
              pxscl=None, pxscl1=None, pxscl2=None,
              lognorm1=False, lognorm2=False,
              vmin1=None, vmax1=None, vmin2=None, vmax2=None, 
              patches1=None, patches2=None,
              display_fig=True, 
              return_fig=False, 
              figsize=(10,4), dpi=125, wspace=0.45):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    if n is not None:
        arr1 = trim(arr1, n)
        arr2 = trim(arr2, n)
    
    if pxscl is not None and pxscl!=np.nan and pxscl!=np.inf:
        vext = pxscl.value * arr1.shape[0]/2
        hext = pxscl.value * arr1.shape[1]/2
        extent1 = [-vext,vext,-hext,hext]
        
        vext = pxscl.value * arr2.shape[0]/2
        hext = pxscl.value * arr2.shape[1]/2
        extent2 = [-vext,vext,-hext,hext]
#         print(pxscl, extent1, extent2)
        if pxscl.unit.is_equivalent(u.meter/u.pix):
            ax[0].set_xlabel('meters')
            ax[1].set_xlabel('meters')
        elif pxscl.unit.is_equivalent(u.arcsec/u.pix):
            ax[0].set_xlabel('arcsec')
            ax[1].set_xlabel('arcsec')
    else:
        if pxscl1 is not None: 
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
            
            if pxscl1.unit.is_equivalent(u.meter/u.pix):
                ax[0].set_xlabel('meters')
            elif pxscl1.unit.is_equivalent(u.arcsec/u.pix):
                ax[0].set_xlabel('arcsec')
        else:
            extent1=None
        if pxscl2 is not None: 
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
            
            if pxscl2.unit.is_equivalent(u.meter/u.pix):
                ax[1].set_xlabel('meters')
            elif pxscl2.unit.is_equivalent(u.arcsec/u.pix):
                ax[1].set_xlabel('arcsec')
        else:
            extent2=None
    
    if lognorm1:
        norm1 = LogNorm(vmin=vmin1,vmax=vmax1)
    else:
        norm1 = Normalize(vmin=vmin1,vmax=vmax1)
        
    if lognorm2:
        norm2 = LogNorm(vmin=vmin2,vmax=vmax2)
    else:
        norm2 = Normalize(vmin=vmin2,vmax=vmax2)
        
    # first plot
    im = ax[0].imshow(arr1, cmap='magma', norm=norm1, extent=extent1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap='magma', norm=norm2, extent=extent2)
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

def disp_wave(wf, n=None, name=None, 
              patches1=None, patches2=None,
              figsize=(10,4), dpi=125, wspace=0.35):
    
    wave = wf.wavefront
    
    pxscl = wf.pixelscale
    if n:
        wave = trim(wave, n)
        
    if name:
        title1 = name + ' Intensity'
        title2 = name + ' Phase'
    else:
        title1 = None
        title2 = None

    myimshow2(np.abs(wave)**2, np.angle(wave),
              title1=title1, title2=title2,
              pxscl=pxscl,
              norm1=LogNorm(), 
              patches1=patches1, patches2=patches2,
              figsize=figsize, dpi=dpi, wspace=wspace)
    
def disp_waves(wfs, wfnames=None, figsize=(10,4), dpi=125, wspace=0.35):
    for wfnum,wf in enumerate(wfs):
        if wfnames:
            wfname = wfnames[wfnum]
        else:
            wfname = None

        disp_wave(wfs[wfnum], name=wfname)
        
        
def mft2( field_in, dout, D, nout, direction, xoffset=0, yoffset=0, xc=0, yc=0 ):

    nfield_in = field_in.shape[1] 
    nfield_out = int(nout)
 
    x = (np.arange(nfield_in) - nfield_in//2 - xc) 
    y = (np.arange(nfield_in) - nfield_in//2 - yc) 

    u = (np.arange(nfield_out) - nfield_out//2 - xoffset/dout) * (dout/D)
    v = (np.arange(nfield_out) - nfield_out//2 - yoffset/dout) * (dout/D)

    xu = np.outer(x, u)
    yv = np.outer(y, v)

    if direction == -1:
        expxu = dout/D * np.exp(-2.0 * np.pi * -1j * xu)
        expyv = np.exp(-2.0 * np.pi * -1j * yv).T
    else:
        expxu = dout/D * np.exp(-2.0 * np.pi * 1j * xu)
        expyv = np.exp(-2.0 * np.pi * 1j * yv).T

    t1 = np.dot(expyv, field_in)
    t2 = np.dot(t1, expxu)

    return t2

def ffts( wavefront, direction ):
    if wavefront.dtype != 'complex128' and wavefront.dtype != 'complex64':
        wavefront = wavefront.astype(complex)

    n = wavefront.shape[0]  # assumed to be power of 2
    wavefront[:,:] = np.roll( np.roll(wavefront, -n//2, 0), -n//2, 1 )  # shift to corner
    
    if proper.use_fftw:
        proper.prop_load_fftw_wisdom( n, proper.fftw_multi_nthreads ) 
        if direction == -1:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_FORWARD' ) 
            wavefront /= np.size(wavefront)
        else:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_BACKWARD' ) 
            wavefront *= np.size(wavefront)
    else:
        if direction == -1:
            wavefront[:,:] = np.fft.fft2(wavefront) / np.size(wavefront)
        else:
            wavefront[:,:] = np.fft.ifft2(wavefront) * np.size(wavefront)
    
    wavefront[:,:] = np.roll( np.roll(wavefront, n//2, 0), n//2, 1 )    # shift to center 

    return wavefront

def trim( input_image, output_dim ):

    input_dim = input_image.shape[1]

    if input_dim == output_dim:
        return input_image
    elif output_dim < input_dim:
        x1 = input_dim // 2 - output_dim // 2
        x2 = x1 + output_dim
        output_image = input_image[x1:x2,x1:x2].copy()
    else:
        output_image = np.zeros((output_dim,output_dim), dtype=input_image.dtype)
        x1 = output_dim // 2 - input_dim // 2
        x2 = x1 + input_dim
        output_image[x1:x2,x1:x2] = input_image

    return output_image

def pol2rect(amp, phs):
    return amp * np.exp(1j*phs)

def rect2pol(x):
    return abs(x), np.angle(x)


        
        
        
        
        
        
        
        
        
        