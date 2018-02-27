import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pylab
import utils
import os

# load metadata and features.
tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')

rockGenreIds = []
hiphopGenreIds = []
popGenreIds = []
for key in tracks['track','genres'].keys():
    if len(tracks['track','genres'][key]) >= 1:
        if tracks['track','genres'][key][0] == 12:
            rockGenreIds.append(key)
        if tracks['track','genres'][key][0] == 21:
            hiphopGenreIds.append(key)
        if tracks['track','genres'][key][0] == 10:
            popGenreIds.append(key)
# print ("example of rock genre tracks")
# print (rockGenreIds[0:1000])
# print ("example of hiphop genre tracks")
# print (hiphopGenreIds[0:1000])
# print ("example of pop genre tracks")
# print (popGenreIds[0:1000])

rockGenreIds = rockGenreIds[0:1000]
hiphopGenreIds = hiphopGenreIds[0:1000]
popGenreIds = popGenreIds[0:1000]
audioDirectory = "data/fma_small/"
spectroDirectory = "data/fma_small_spectro/"
for rockId in rockGenreIds:
    rockId = str(rockId)
    while(len(rockId) < 6):
        rockId = "0" + rockId
    #print (rockId)
    #create dir if it doesn't exist
    if not os.path.exists(spectroDirectory+rockId[0:3]):
        os.makedirs(spectroDirectory+rockId[0:3])
    filename = rockId[0:3]+"/"+rockId
    audioFilename = audioDirectory + filename + ".mp3"
    #print (audioFilename)
    spectroFilename = spectroDirectory + filename + ".jpg"
    #print (spectroFilename)
    if os.path.isfile(audioFilename):
        pylab.figure()
        x, sr = librosa.load(audioFilename, sr=None, mono=True)
        #stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        #mel = librosa.core.stft(sr=sr, S=stft**2)
        spectro = librosa.core.stft(x)

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(librosa.power_to_db(spectro, ref=np.max))
        pylab.savefig(spectroFilename, bbox_inches=None, pad_inches=0)
        pylab.close()
        
for hiphopId in hiphopGenreIds:
    hiphopId = str(hiphopId)
    while(len(hiphopId) < 6):
        hiphopId = "0" + hiphopId
    #print (rockId)
    #create dir if it doesn't exist
    if not os.path.exists(spectroDirectory+hiphopId[0:3]):
        os.makedirs(spectroDirectory+hiphopId[0:3])
    filename = hiphopId[0:3]+"/"+hiphopId
    audioFilename = audioDirectory + filename + ".mp3"
    #print (audioFilename)
    spectroFilename = spectroDirectory + filename + ".jpg"
    #print (spectroFilename)
    if os.path.isfile(audioFilename):
        pylab.figure() 
        x, sr = librosa.load(audioFilename, sr=None, mono=True)
        #stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        #mel = librosa.core.stft(sr=sr, S=stft**2)
        spectro = librosa.core.stft(x)

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(librosa.power_to_db(spectro, ref=np.max))
        pylab.savefig(spectroFilename, bbox_inches=None, pad_inches=0)
        pylab.close()
        
for popId in popGenreIds:
    popId = str(popId)
    while(len(popId) < 6):
        popId = "0" + popId
    #print (rockId)
    #create dir if it doesn't exist
    if not os.path.exists(spectroDirectory+popId[0:3]):
        os.makedirs(spectroDirectory+popId[0:3])
    filename = popId[0:3]+"/"+popId
    audioFilename = audioDirectory + filename + ".mp3"
    #print (audioFilename)
    spectroFilename = spectroDirectory + filename + ".jpg"
    #print (spectroFilename)
    if os.path.isfile(audioFilename):
        pylab.figure() 
        x, sr = librosa.load(audioFilename, sr=None, mono=True)
        #stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        #mel = librosa.core.stft(sr=sr, S=stft**2)
        spectro = librosa.core.stft(x)

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(librosa.power_to_db(spectro, ref=np.max))
        pylab.savefig(spectroFilename, bbox_inches=None, pad_inches=0)
        pylab.close()