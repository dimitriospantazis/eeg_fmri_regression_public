import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr


def merge_eeg_fmri(eegfile,fmrifile_dmn,fmrifile_dan, offset=7, sfreq=150):
    # Adds the fmri time courses to the eeg raw file as additional misc channels. 
    # It uses cubic splines to interpolate the fmri data to the eeg sampling rate
    # It also shifts the fmri data by offset seconds (e.g 7 seconds). If offset>0, the fmri data are shifted to the left 
    # (earlier in time). If offset<0, the fmri data are shifted to the right (delayed in time)
    # Most likely, eeg data will be able to predict future fmri values (thus, use offset>0)
    
    # Load the eeg .set file
    raw = mne.io.read_raw_eeglab(eegfile, preload=True)
    
    # Load and interpolate fmri data to EEG time course
    x_dmn = interpolate_fmri2eeg(raw, np.loadtxt(fmrifile_dmn))
    x_dmn /= np.nanstd(x_dmn) #normalize with std excluding nan values  
    x_dan = interpolate_fmri2eeg(raw, np.loadtxt(fmrifile_dan))
    x_dan /= np.nanstd(x_dan) #normalize with std excluding nan values  
    miscdata = np.vstack((x_dmn, x_dan))

    # Offset fMRI data relative to EEG 
    dt_samples = np.round(raw.info['sfreq']*offset).astype(int) #offset samples
    miscdata_offset = np.full_like(miscdata, np.nan, dtype=np.float64) # Create an array of NaNs
    if offset > 0: #if shift fMRI earlier in time by offset seconds
        miscdata_offset[:,:-dt_samples] = miscdata[:,dt_samples:] #shift to the left       
    elif offset < 0:
        miscdata_offset[:,-dt_samples:] = miscdata[:,:dt_samples] #shift to the right
    else:
        miscdata_offset = miscdata

   # add fmri time course as misc channels to raw
    misc_info = mne.create_info(ch_names=['fMRI DMN', 'fMRI DAN'], sfreq=raw.info['sfreq'], ch_types=['misc']*2)
    raw_misc = mne.io.RawArray(data = miscdata_offset, info = misc_info)
    raw.add_channels([raw_misc])

    # crop raw to remove nan values from start and end
    t_idx = np.arange(0,raw.n_times)
    t_idx_no_nan = t_idx[~np.isnan(miscdata_offset[0,:])]
    raw.crop(tmin=min(t_idx_no_nan)/raw.info['sfreq'], tmax=max(t_idx_no_nan)/raw.info['sfreq'])

    # resample data to lower sampling frequency, 150Hz
    raw.resample(150, npad='auto')

    return raw


def interpolate_fmri2eeg(raw, fmridata):
    # Interpolate fmri data to EEG time course

    # EEG T1 events
    events, event_id = mne.events_from_annotations(raw)
    idx_T1 = events[events[:,2]==event_id['T  1'],0]
    # Cubic spline interpolation
    indices = idx_T1[:-1] #exclude last value that was not recorded in fmri
    values = fmridata
    interpolator = CubicSpline(indices, values, extrapolate=False)
    x = interpolator(range(raw.n_times))
    #plt.plot(indices,values,'o')
    #plt.plot(x)
    #plt.show()
    return x







