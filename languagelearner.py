from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SequentialDataSet
from pybrain.structure import SigmoidLayer
from pybrain.structure import LSTMLayer
from pybrain.structure import LinearLayer
from pybrain.structure import RecurrentNetwork
from pybrain.structure import FullConnection

import itertools
import numpy as np
import random

class LanguageLearner:

	__OUTPUT = "Sample at {0} epochs (prompt=\"{1}\", length={2}): {3}"

	def __init__(self, trainingText, hiddenLayers, hiddenNodes):
		self.__initialized = False
		with open(trainingText) as f:
			self.raw = f.read()
		self.characters = list(self.raw)
		self.rawData = list(map(ord, self.characters))
		print("Creating alphabet mapping...")
		self.mapping = []
		for charCode in self.rawData:
			if charCode not in self.mapping:
				self.mapping.append(charCode)
		print("Mapping of " + str(len(self.mapping)) + " created.")
		print(str(self.mapping))
		print("Converting data to mapping...")
		self.data = []
		for charCode in self.rawData:
			self.data.append(self.mapping.index(charCode))
		print("Done.")
		self.dataIn = self.data[:-1:]
		self.dataOut = self.data[1::]
		self.inputs = 1
		self.hiddenLayers = hiddenLayers
		self.hiddenNodes = hiddenNodes
		self.outputs = 1

	def initialize(self, verbose):
		print("Initializing language learner...")
		self.verbose = verbose

		# Create network and modules
		self.net = RecurrentNetwork()
		inp = LinearLayer(self.inputs, name="in")
		hiddenModules = []
		for i in range(0, self.hiddenLayers):
			hiddenModules.append(LSTMLayer(self.hiddenNodes, name=("hidden-" + str(i + 1))))
		outp = LinearLayer(self.outputs, name="out")

		# Add modules to the network with recurrence
		self.net.addOutputModule(outp)
		self.net.addInputModule(inp)
		
		for module in hiddenModules:
			self.net.addModule(module)

		# Create connections

		self.net.addConnection(FullConnection(self.net["in"], self.net["hidden-1"]))
		for i in range(0, len(hiddenModules) - 1):
			self.net.addConnection(FullConnection(self.net["hidden-" + str(i + 1)], self.net["hidden-" + str(i + 2)]))
			self.net.addRecurrentConnection(FullConnection(self.net["hidden-" + str(i + 1)], self.net["hidden-" + str(i + 1)]))
		self.net.addRecurrentConnection(FullConnection(self.net["hidden-" + str(len(hiddenModules))],
			self.net["hidden-" + str(len(hiddenModules))]))
		self.net.addConnection(FullConnection(self.net["hidden-" + str(len(hiddenModules))], self.net["out"]))
		self.net.sortModules()

		self.trainingSet = SequentialDataSet(self.inputs, self.outputs)
		for x, y in zip(self.dataIn, self.dataOut):
			self.trainingSet.newSequence()
			self.trainingSet.appendLinked([x], [y])

		self.net.randomize()

		print("Neural network initialzed with structure:")
		print(self.net)

		self.trainer = BackpropTrainer(self.net, self.trainingSet, verbose=verbose)
		self.__initialized = True
		print("Successfully initialized network.")

	def train(self, epochs, frequency, prompt, length):
		if not self.__initialized:
			raise Exception("Attempted to train uninitialized LanguageLearner")
		print ("Beginning training for " + str(epochs) + " epochs...")
		if frequency >= 0:
			print(LanguageLearner.__OUTPUT.format(0, prompt, length, self.sample(prompt, length)))
		for i in range(1, epochs):
			print("Error at " + str(i) + " epochs: " + str(self.trainer.train()))
			if i % frequency == 0:
				print(LanguageLearner.__OUTPUT.format(i, prompt, length, self.sample(prompt, length)))
		print("Completed training.")

	def sample(self, prompt, length):
		self.net.reset()
		if prompt == None:
			prompt = chr(random.choice(self.mapping))
		output = prompt
		charCode = ord(prompt)
		for i in range(0, length):
			sampledResult = self.net.activate([charCode])
			charCode = int(round(sampledResult[0]))
			if charCode < 0 or charCode >= len(self.mapping):
				return output + "#TERMINATED_SAMPLE(reason: learner guessed invalid character)"
			output += chr(self.mapping[charCode])
		return output
