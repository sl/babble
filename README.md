Babble
======

Babble is a recurrent neural network (more particularly LSTM) based character level language modeling program. In essence, given some plaintext source it can learn the language behind it!

Switch to PyThink
-----------------

Babble is officially switching away from Pybrain. While Pybrain is an amazing library, it currently doesn't support hardware acceleration, and is limited by the speed of the python interpreter. Due to this limitation, we're switching to an in house neural network library (named PyThink) which will be built using anaconda to minimize training times.

The pybrain implementation is currently being deprecated in favor of a completely home build implementation which executed on the GPU. The source for it can still be found in the pybrain folder though.

