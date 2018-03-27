import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pylab
import utils_spec
import os

# Load metadata and features.
tracks = utils_spec.load('/ubc/cs/research/tracking-raid/candice/project/DreamingInMusic/data/fma_metadata/tracks.csv')
genres = utils_spec.load('/ubc/cs/research/tracking-raid/candice/project/DreamingInMusic/data/fma_metadata/genres.csv')

rockGenreIds = []
hiphopGenreIds = []
popGenreIds = []
jazzGenreIds = []
elecGenreIds = []
classicalGenreIds = []
for key in tracks['track','genres'].keys():
    if len(tracks['track','genres'][key]) >= 1:
        if tracks['track','genres'][key][0] == 12:
            rockGenreIds.append(key)
        if tracks['track','genres'][key][0] == 21:
            hiphopGenreIds.append(key)
        if tracks['track','genres'][key][0] == 10:
            popGenreIds.append(key)
        if tracks['track','genres'][key][0] == 4:
            jazzGenreIds.append(key)
        if tracks['track','genres'][key][0] == 15:
            elecGenreIds.append(key)
        if tracks['track','genres'][key][0] == 5:
            classicalGenreIds.append(key)
            
def get_filenames(audioDirectory, ids):
    file_list = []
    for count, Id in enumerate(ids):
        try:
            Id = ids[count]
            #print count
            Id = str(Id)
            while(len(Id) < 6):
                Id = "0" + Id
            #print (rockId)
            path = os.path.join(audioDirectory, Id[0:3])
            files = os.listdir(path)
            audioFilename = os.path.join(path, Id + ".mp3")
            if os.path.isfile(audioFilename):
                file_list.append(audioFilename)
        except:
            print count, Id
        
    return file_list

audioDirectory = "/ubc/cs/research/tracking-raid/candice/project/fma_medium"
rock_song_list = get_filenames(audioDirectory, rockGenreIds)
hiphop_song_list = get_filenames(audioDirectory, hiphopGenreIds)
elec_song_list = get_filenames(audioDirectory, elecGenreIds)
classical_song_list = get_filenames(audioDirectory, classicalGenreIds)

print len(rock_song_list)
print len(hiphop_song_list)
elec_song_list = elec_song_list[:2000]
print len(elec_song_list)
print len(classical_song_list)
song_list = rock_song_list + hiphop_song_list + elec_song_list + classical_song_list
print len(song_list)

import scipy.misc

minAmp = -80.0
maxAmp = 1.9073486e-06
def scaleSpectro(x, new_size):
    #print ("before scaling", x)
    x = (x - minAmp)/(maxAmp - minAmp)
    #print ("after scaling", x)
    y = scipy.misc.imresize(x, new_size, mode='L', interp='nearest')
    return y
def unscaleSpectro(x, new_size, minAmp, maxAmp):
    x = scipy.misc.imresize(x, new_size, mode='L', interp='nearest')
    x = x/255.0
    x = minAmp + x*(maxAmp - minAmp)
    return x
def convert_file_to_spectro(audio_filename):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    try:
        x1, sr1 = librosa.load(audio_filename, sr=None, mono=True, duration=9.98, offset=0)
    except:
        return
    try:
            x2, sr2 = librosa.load(audio_filename, sr=None, mono=True, duration=9.98, offset=10)
    except:
        return
    try:
        x3, sr3 = librosa.load(audio_filename, sr=None, mono=True, duration=9.98, offset=20)
    except:
        return
    #Convert audio to a complex valued spectrogram
    spectro1 = librosa.core.stft(x1)
    spectro2 = librosa.core.stft(x2)
    spectro3 = librosa.core.stft(x3)
    if spectro1 is not None and spectro2 is not None and spectro3 is not None:
        #Separate out amplitude and phase from complex valued spectrogram
        mag1, phase1 = librosa.core.magphase(spectro1)
        mag2, phase2 = librosa.core.magphase(spectro2)
        mag3, phase3 = librosa.core.magphase(spectro3)

        #Get the decibal version from power spectrogram
        #This is the value that should be stored for training
        powerToDb1 = librosa.power_to_db(mag1, ref=np.max)
        powerToDb2 = librosa.power_to_db(mag2, ref=np.max)
        powerToDb3 = librosa.power_to_db(mag3, ref=np.max)
        return powerToDb1, powerToDb2, powerToDb3
    else:
        return None, None, None
    
def save_spectros(song_list, genre_name, save_dir):
    num_songs = len(song_list)
    num_val_songs = num_songs / 5
    print num_songs, num_val_songs
    #for i, song in enumerate(song_list):
    #    print i, song
    for i, song in enumerate(song_list):
        print i, song
        if i < num_val_songs:
            #print "val"
            save_ddir = os.path.join(save_dir, 'val')
        else:
            #print "train"
            save_ddir = os.path.join(save_dir, 'train')
        if not os.path.exists(save_ddir):
            os.makedirs(save_ddir)
        save_subdir = os.path.join(save_ddir, genre_name)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        if convert_file_to_spectro(song) is not None:
            spectro1, spectro2, spectro3 = convert_file_to_spectro(song)
        else:
            continue
        if spectro1 is not None and spectro2 is not None and spectro3 is not None:
            scaled_spectro1 = scaleSpectro(spectro1, (512, 512))
            scaled_spectro2 = scaleSpectro(spectro2, (512, 512))
            scaled_spectro3 = scaleSpectro(spectro3, (512, 512))
            save_name1 = os.path.join(save_subdir, str(3*i+1) + '.jpg')
            scipy.misc.imsave(save_name1, scaled_spectro1)
            save_name2 = os.path.join(save_subdir, str(3*i+2) + '.jpg')
            scipy.misc.imsave(save_name2, scaled_spectro2)
            save_name3 = os.path.join(save_subdir, str(3*i+3) + '.jpg')
            scipy.misc.imsave(save_name3, scaled_spectro3)
            
save_dir = '/ubc/cs/research/tracking-raid/candice/project/dataset/mag_512'
#save_spectros(rock_song_list, 'rock', save_dir)
save_spectros(hiphop_song_list[1158:], 'hiphop', save_dir)
#save_spectros(classical_song_list, 'classical', save_dir)
















