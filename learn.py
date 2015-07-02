from babble import LanguageLearner

import sys

presets = {
	"voynich": "res/voynich/eva-raw.eva",
	"voynich-short": "res/voynich/eva-raw-shortened.eva",
	"voynich-debug": "res/voynich/eva-raw-debug.eva",
	"mobydick": "res/english/mobydick.txt",
	"mobydick-debug": "res/english/mobydick-debug.txt"
}

def error(msg):
	print("Error: " + msg + ", use \"python learn.py -h\" (without quotes) for help")
	sys.exit(0)

rawArgs = sys.argv[1:]
flags = list(map((lambda raw : raw[1:]), filter((lambda s : s.startswith("-")), rawArgs)))
cmdArgs = list(filter((lambda s : not s.startswith("-")), rawArgs))

if "h" in flags or "help" in flags:
	print("usage: python learn.py [-h | -help] [-p | -preset] <name> <epochs> [layers]")
	print("           [layer-size] [sample-frequency] [sample-length] [prompt]")
	print()
	print("-p | -preset      : use a preset name rather than a specific file")
	print("-v | -verbose     : run in verbose mode")
	print("name (required)   : the file or preset to train on")
	print("epochs (required) : the amount of training epochs")
	print("layers            : the amount of hidden layers to use for the nerual net")
	print("layer-size        : the amount of neurons per hidden layer to use for the neural net")
	print("sample-frequency  : the frequency with which to sample, defaults to 1, -1 to disable sampling")
	print("prompt            : the word or character to prompt the neural nets generation with")
	sys.exit(0)

settings = {}
# try:
if "p" in flags or "preset" in flags:
	settings["trainingfile"] = presets[cmdArgs[0]]
else:
	settings["trainingfile"] = cmdArgs[0]

settings["epochs"] = int(float(cmdArgs[1]))

# Default Settings
settings["verbose"] = False
settings["layers"] = 2
settings["layersize"] = 100
settings["samplefrequency"] = 1
settings["samplelength"] = 15
settings["prompt"] = None

if "v" in flags or "verbose" in flags:
		settings["verbose"] = True

if len(cmdArgs) >= 3:
	settings["layers"] = int(float(cmdArgs[2]))
if len(cmdArgs) >= 4:
	settings["layersize"] = int(float(cmdArgs[3]))
if len(cmdArgs) >= 5:
	settings["samplefrequency"] = int(float(cmdArgs[4]))
if len(cmdArgs) >= 6:
	settings["samplelength"] = int(float(cmdArgs[5]))
if len(cmdArgs) >= 7:
	settings["prompt"] = cmdArgs[6]

# except:
# 	error("Invalid arguments")

learner = LanguageLearner(settings["trainingfile"], settings["layers"], settings["layersize"])
learner.initialize(settings["verbose"])
input("Press enter to begin training")
learner.train(settings["epochs"], settings["samplefrequency"], settings["prompt"], settings["samplelength"])






# learner = LanguageLearner("res/voyinch/eva-raw-shortened.eva", 2, 100)
# learner.initialize(True)
# input("Press enter to begin training")
# learner.train(100, 1, None, 15)