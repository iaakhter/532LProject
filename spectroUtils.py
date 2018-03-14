import os
import utils
import librosa
import numpy as np
import IPython.display as ipd
import scipy.misc
import torch
from torch.autograd import Variable


# Load metadata and features.
tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')
audioDirectory = "data/fma_small/"
spectroSize = 224

# find min and mix amplitude of spectro grams
def minMax():
	directory = "data/fma_small"

	minAmp = float("inf")
	maxAmp = -float("inf")
	for subdir, dirs, files in os.walk(directory):
		for file in files:
			filename = os.fsdecode(file)
			if filename.endswith(".mp3"): 
				#print (os.path.join(subdir, file))
				audioFilename = os.path.join(subdir, file)

				x, sr = librosa.load(audioFilename, sr=None, mono=True)

				#Convert audio to a complex valued spectrogram
				spectro = librosa.core.stft(x)

				#Separate out amplitude and phase from complex valued spectrogram
				mag, phase = librosa.core.magphase(spectro)
				#print ("mag", mag)
				#print ("phase",phase)

				#Get the decibal version from power spectrogram
				#This is the value that should be stored for training
				powerToDb = librosa.power_to_db(mag, ref=np.max)
				
				locMin = np.amin(powerToDb)
				locMax = np.amax(powerToDb)
				minAmp = min(minAmp, locMin)
				maxAmp = max(maxAmp, locMax)
				
	print ("minAmp", minAmp)
	print ("maxAmp", maxAmp)
	return (minAmp, maxAmp)

def scaleSpectro(x, new_size, minAmp, maxAmp):
	x = (x - minAmp)/(maxAmp - minAmp)
	#print ("after scaling", x)
	y = scipy.misc.imresize(x, new_size, mode='L', interp='nearest')
	return y

def unscaleSpectro(x, new_size, minAmp, maxAmp):
	x = scipy.misc.imresize(x, new_size, mode='L', interp='nearest')
	x = x/255.0
	x = minAmp + x*(maxAmp - minAmp)
	x = librosa.core.db_to_power(x)
	return x

# return the first genreId of a track (a track can have multiple genres)
# rock genre id is 12, hiphop is 21 and pop is 10
def getGenreId(trackId):
	if len(tracks['track','genres'][trackId]) >= 1:
		return tracks['track','genres'][trackId][0]
	else:
		# if the track does not have a genre
		return None

def getGenre(trackId):
	if len(tracks['track','genres'][trackId]) >= 1:
		return genres['title'][tracks['track','genres'][trackId][0]]
	else:
		# if the track does not have a genre
		return None

def trackExists(trackIdNum):
	trackId = str(trackIdNum)
	while(len(trackId) < 6):
		trackId = "0" + trackId
	filename = trackId[0:3]+"/"+trackId
	audioFilename = audioDirectory + filename + ".mp3"
	if os.path.isfile(audioFilename):
		return True
	return False

# return scaledSpectrogram, original phase and original spectrogram shape
def loadSpectro(trackIdNum, spectroSize, minAmp, maxAmp, trackDuration = 10, startTime = 0):
	trackId = str(trackIdNum)
	while(len(trackId) < 6):
		trackId = "0" + trackId
	filename = trackId[0:3]+"/"+trackId
	audioFilename = audioDirectory + filename + ".mp3"
	if trackExists(trackIdNum):
		x, sr = librosa.load(audioFilename, sr=None, mono=True, duration = trackDuration, offset=startTime)
		#Convert audio to a complex valued spectrogram
		spectro = librosa.core.stft(x)

		#Separate out amplitude and phase from complex valued spectrogram
		mag, phase = librosa.core.magphase(spectro)
		#print ("mag", mag)
		#print ("phase",phase)

		#Get the decibal version from power spectrogram
		#This is the value that should be stored for training
		powerToDb = librosa.power_to_db(mag, ref=np.max)
		scaledSpectro = scaleSpectro(powerToDb, (spectroSize, spectroSize), minAmp, maxAmp)
		spectroTensor = torch.from_numpy(scaledSpectro).type(torch.DoubleTensor)
		spectroVar = Variable(spectroTensor).unsqueeze(0)
		return spectroVar, phase, powerToDb.shape
	else:
		print("Audio does not exist")
		return None