
"""
Simple pitch estimation
"""
from __future__ import print_function
import sys
import time
import os
import glob
import aifc
import math
import os
import numpy 
from scipy import signal
import scipy as sp
from scipy.io import wavfile
from scipy.signal import correlate
import matplotlib.pyplot as plt
from numpy import NaN, Inf, arange, isscalar, array
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy import linalg as la
from python_speech_features import mfcc
from python_speech_features import sigproc
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import ShuffleSplit
import random

eps = 0.00000001
__author__ = "Jose A. R. Fonollosa"

def ZCR(frame):
	"""Computes zero crossing rate of frame"""
	count = len(frame)
	countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
	return (numpy.float64(countZ) / numpy.float64(count-1.0))


def Energy(frame):
	"""Computes signal energy of frame"""
	return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def mfccInitFilterBanks(fs, nfft):
	"""
	Computes the triangular filterbank for MFCC computation (used in the stFeatureExtraction function before the stMFCC function call)
	This function is taken from the scikits.talkbox library (MIT Licence):
	https://pypi.python.org/pypi/scikits.talkbox
	"""

	# filter bank params:
	lowfreq = 133.33
	linsc = 200/3.
	logsc = 1.0711703
	numLinFiltTotal = 13
	numLogFilt = 27

	if fs < 8000:
		nlogfil = 5

	# Total number of filters
	nFiltTotal = numLinFiltTotal + numLogFilt

	# Compute frequency points of the triangle:
	freqs = numpy.zeros(nFiltTotal+2)
	freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
	freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
	heights = 2./(freqs[2:] - freqs[0:-2])

	# Compute filterbank coeff (in fft domain, in bins)
	fbank = numpy.zeros((nFiltTotal, nfft))
	nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

	for i in range(nFiltTotal):
		lowTrFreq = freqs[i]
		cenTrFreq = freqs[i+1]
		highTrFreq = freqs[i+2]

		lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
		lslope = heights[i] / (cenTrFreq - lowTrFreq)
		rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
		rslope = heights[i] / (highTrFreq - cenTrFreq)
		fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
		fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

	return fbank, freqs   

def MFCC(X, fbank, nceps):
	"""
	Computes the MFCCs of a frame, given the fft mag

	ARGUMENTS:
		X:        fft magnitude abs(FFT)
		fbank:    filter bank (see mfccInitFilterBanks)
	RETURN
		ceps:     MFCCs (13 element vector)

	Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
	#    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
	"""

	mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
	ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
	return ceps

def autocorr_feat(frame):

	defvalue = (0.0, 1.0)
	corr = correlate(frame, frame)
	# keep the positive part
	corr = corr[round(len(corr)/2):]

	# Find the first minimum
	dcorr = numpy.diff(corr)
	rmin = numpy.where(dcorr > 0)[0]
	if len(rmin) > 0:
		rmin1 = rmin[0]
	else:
		return defvalue

	# Find the next peak
	peak = numpy.argmax(corr[rmin1:]) + rmin1

	# Two features
	r1=corr[1]/corr[0]
	rmax = corr[peak]/corr[0]

	return r1,rmax
	

def autocorr_method(frame, rate):
	"""Estimate pitch using autocorrelation
	"""
	defvalue = (0.0, 1.0)

	# Calculate autocorrelation using scipy correlate
	frame = frame.astype(numpy.float)
	frame -= frame.mean()
	amax = numpy.abs(frame).max()
	if amax > 0:
		frame /= amax
	else:
		return defvalue

	corr = correlate(frame, frame)
	# keep the positive part
	corr = corr[round(len(corr)/2):]

	# Find the first minimum
	dcorr = numpy.diff(corr)
	rmin = numpy.where(dcorr > 0)[0]
	if len(rmin) > 0:
		rmin1 = rmin[0]
	else:
		return defvalue

	# Find the next peak
	peak = numpy.argmax(corr[rmin1:]) + rmin1
	rmax = corr[peak]/corr[0]
	f0 = rate / peak

	if f0 > 50 and f0 < 550:
		return f0
	else:
		return f0

def ComputeCepstrum(frame,rate):

	N=len(frame)
	epsilon=0.0001

	windowed = frame * signal.hamming(N)
	spectrum = numpy.fft.rfft(windowed)
	spectrum[spectrum == 0] = epsilon
	log_spectrum = numpy.log(numpy.abs(spectrum**2))
	ceps = numpy.fft.irfft(log_spectrum)

	return ceps

def CepstrumBased_PitchExtraction(frame,rate):

	defvalue = (0.0, 1.0)
	# Calculate Cepstrum of speech frame
	frame = frame.astype(numpy.float)
	frame -= frame.mean()
	amax = numpy.abs(frame).max()
	if amax > 0:
		frame /= amax
	else:
		return defvalue

	ceps=ComputeCepstrum(frame,rate) # In time , c(z)

	# To discriminate between vocal tract and excitation source, we define a  range to look for the peak corresponding to the fund. frequency

	LowFreq=50
	HighFreq=550

	start=int(rate/HighFreq)
	end=int(rate/LowFreq)

	ceps-=ceps.mean()
	maxceps=numpy.abs(ceps).max()
	ceps/=maxceps

	i_peak   = numpy.argmax(ceps[start:end])
	#i_interp = scipy.interpolate(ceps[start:end+1], i_peak)[0]
	f0= rate / (i_peak + start)


	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# line, = ax.plot(ceps, lw=2)
	# plt.axvline(x=start)
	# plt.axvline(x=end)
	# ax.annotate('Pitch peak', xy=(i_peak + start, ceps[i_peak + start]),arrowprops=dict(facecolor='black', shrink=0.05),)
	# plt.show()

	#Th_voiced= ceps[i_peak+start]/maxceps

	if  f0 > 50 and f0 < 550:
		return f0
	else:
		return 0;



def wav2f0(options, gui):

	with open(gui) as f:
		for line in f:
			line = line.strip()
			if len(line) == 0:
				continue
			filename = os.path.join(options.datadir, line + ".wav")
			f0_filename = os.path.join(options.datadir, line + ".f0")
			print("Processing:", filename, '->', f0_filename)
			rate, data = wavfile.read(filename)
			with open(f0_filename, 'wt') as f0file:
				nsamples = len(data)
				# From miliseconds to samples
				frame_len = int(round((options.windowlength * rate) / 1000))
				frame_step= int(round((options.framelength * rate) / 1000))
				for ini in range(0, nsamples - frame_len + 1, frame_step):
					frame = data[ini:ini+frame_len]
					f0 = autocorr_method(frame, rate)
					print(f0, file=f0file)


def FeatureExtraction(signal,rate,Win,Step):

	# Already in samples
	Win = int(Win)
	Step = int(Step)

	# Signal normalization
	signal = numpy.double(signal)

	signal = signal / (2.0 ** 15) ## ???
	DC = signal.mean()
	MAX = (numpy.abs(signal)).max()
	signal = (signal - DC) / MAX

	N = len(signal) # total number of samples
	curPos = 0
	nFFT = round(Win/2)

	# Compute the triangular filter banks used in the mfcc calculation
	[fbank, freqs] = mfccInitFilterBanks(rate, nFFT) 


	Features=[]
	nceps = 13
	totalNumOfFeatures=nceps

	while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
		x = signal[curPos:curPos+Win]                    # get current window	
		curPos = curPos + Step                           # update window position
		X = abs(fft(x))                                  # get fft magnitude
		X = X[0:nFFT]                                    # normalize fft
		X = X / len(X)

		curFV = numpy.zeros((totalNumOfFeatures, 1))

		#curFV[0] = ZCR(x)                                # zero crossing rate
		#curFV[1] = numpy.log10(Energy(x)) 					#log-energy of the frame
		#[r1,rmax] = autocorr_feat(x) 
		#curFV[2] = r1                                    # Correlation coef at sample 1
		#curFV[3] = rmax                     		     # Correlation coef at maximum pitch peak
		curFV[0:nceps,0] = MFCC(X, fbank, nceps)  # MFCCs
		Features.append(curFV)
		
	Features = numpy.concatenate(Features, 1)
	Features = numpy.transpose(Features)
	return Features


def main(options):

	# Selected Number of Audio Files from Database
	Train_Files=250
	TrainLabels=[]
	X_train=[]
	Training_Dataset="ptdb_tug.gui"

	with open(Training_Dataset) as f:
		lines=f.readlines()
		random.shuffle(lines)
		for num,line in enumerate(lines):
			line = line.strip()
			if len(line) == 0:
				continue
			filename = os.path.join(options.datadir, line + ".wav")
			f0ref_filename = os.path.join(options.datadir, line +".f0ref")
			print("Extracting Features from Training File#",num,": ", filename)
			rate, data = wavfile.read(filename)

			if num == Train_Files:
				break
			else:
				# Generate  Target Labels
				with open(f0ref_filename) as f0ref:
					lines=f0ref.readlines()
					lines = [x.strip() for x in lines] 
					#lines = (line for line in lines if line) # Non-blank lines
					lines = [float(i) for i in lines]
					Label=numpy.array(lines)
					Label[ Label > 0] = 1
					TrainLabels.append(Label)

				# FEATURE EXTRACTION OF AUDIO FRAMES
				Win = int(round((0.032 * rate)))
				Step= int(round((0.010 * rate)))
				X_train.append(FeatureExtraction(data,rate,Win,Step)[0:len(Label),:])
				#NumFramesTrain=NumFramesTrain+len(X_train)

		TrainLabels=numpy.asarray(TrainLabels)
		X_train=numpy.asarray(X_train)
		TrainLabels=numpy.concatenate(TrainLabels, 0)
		TrainLabels = numpy.transpose(TrainLabels)
		X_train = numpy.concatenate(X_train, 0)

		# PRE-PROCESSING (FEATURE NORMALIZATION)
		scaler = StandardScaler()
		scaler.fit(X_train)  
		X_train = scaler.transform(X_train)
		#X_train = scaler.transform(X_train)
		# TRAINING NEURAL NET
		print("Training Neural Net")
		clf = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(25,10), random_state=1)
		print(X_train.shape)
		clf.fit(X_train,TrainLabels)



		#cv=None
		#plt=plot_learning_curve(clf, "Learning curve", X_train, TrainLabels, ylim=None, cv=cv, n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5))
		#plt.show()


	# Testing/Evaluation  phase 
	TestLabels=[]
	X_test=[]
	f0=[]

	with open("pda_ue.gui") as f:
		files=f.readlines()
		TestLabels=[]
		for line in files:
			line = line.strip()
			if len(line) == 0:
				continue
			filename = os.path.join(options.datadir, line + ".wav")
			f0_filename = os.path.join(options.datadir, line + ".f0")
			f0ref_filename = os.path.join(options.datadir, line +".f0ref")
			print("Extracting Features in Test File:", filename)

			rate, data = wavfile.read(filename)
			# From miliseconds to samples
			Win = int(round((0.032 * rate)))
			Step= int(round((0.015 * rate)))
			curPos=0
			f0=[]
			while (curPos + Win - 1 < len(data)):                # for each short-term window until the end of signal
				x = data[curPos:curPos+Win]                      # get current window	
				curPos = curPos + Step  						 # update window position
				f0.append(CepstrumBased_PitchExtraction(x, rate))                         	

			X=FeatureExtraction(data,rate,Win,Step)
			#X[:,4:]=scaler.transform(X[:,4:])
			X=scaler.transform(X)

			with open(f0_filename, 'wt') as f0file:
				num=0
				for frame in X:
					frame=frame.reshape(1,-1)
					Voiced= clf.predict(frame)
					if Voiced==1 :
						print(f0[num],file=f0file)
					else:
						print(0, file=f0file)
					num+=1	

			X_test.append(X)

			with open(f0ref_filename) as f0ref:
				lines=f0ref.readlines()
				lines = [x.strip() for x in lines] 
				#lines = (line for line in lines if line) # Non-blank lines
				lines = [float(i) for i in lines]
				Label=numpy.array(lines)
				Label[ Label > 0] = 1
				TestLabels.append(Label[:len(X)])

		X_test=numpy.asarray(X_test)
		X_test = numpy.concatenate(X_test, 0)
		TestLabels=numpy.concatenate(TestLabels, 0)
		TestLabels = numpy.transpose(TestLabels)	
		#clf = MLPClassifier(solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(25,10), random_state=1)
		#plt=plot_learning_curve2(clf,"Learning Curve with TRAIN: PTDB_TUG and TEST: FDA_UE",X_train,TrainLabels,X_test,TestLabels)
		#plt.show()




def plot_learning_curve2(estimator, title,X_train,y_train,X_test,y_test):

	plt.figure()
	plt.title(title)
	plt.xlabel("Number of speech frames")
	plt.ylabel("Score")
	plt.grid()
	train_scores, test_scores = [], []
	for n in range(25000, 275000, 25000): # 10 points for the curve
		estimator.fit(X_train[:n,:], y_train[:n])
		train_scores.append(estimator.score(X_train[:n,:], y_train[:n]))
		test_scores.append(estimator.score(X_test, y_test))
	plt.plot(range(25000, 275000, 25000), train_scores,'o-', color="r", label="Training score")
	plt.plot(range(25000, 275000, 25000), test_scores,'o-', color="g",label="Test score")  
	plt.legend(loc="best")
	return plt


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
						n_jobs=1, train_sizes=numpy.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
		An object of that type which is cloned for each validation.

	title : string
		Title for the chart.

	X : array-like, shape (n_samples, n_features)
		Training vector, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
		Target relative to X for classification or regression;
		None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
		Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		  - None, to use the default 3-fold cross-validation,
		  - integer, to specify the number of folds.
		  - An object to be used as a cross-validation generator.
		  - An iterable yielding train/test splits.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : integer, optional
		Number of jobs to run in parallel (default 1).
	"""

	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

	train_scores_mean = numpy.mean(train_scores, axis=1)
	train_scores_std = numpy.std(train_scores, axis=1)
	test_scores_mean = numpy.mean(test_scores, axis=1)
	test_scores_std = numpy.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")
	return plt

if __name__ == "__main__":

	import optparse
	optparser = optparse.OptionParser(
		usage='%prog [OPTION]... FILELIST\n' + __doc__)
	optparser.add_option(
		'-w', '--windowlength', type='float', default=32,
		help='windows length (ms)')
	optparser.add_option(
		'-f', '--framelength', type='float', default=15,
		help='frame shift (ms)')
	optparser.add_option(
		'-d', '--datadir', type='string', default='data',
		help='data folder')

	options, args = optparser.parse_args()
	#wav2f0(options, "pda_ue.gui")
	main(options)
