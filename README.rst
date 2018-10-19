
Transform Learning for Non-Negative Matrix Factorization
--------------------------------------------------------

Transform-learning NMF is a new method for music signal processing. Standard NMF is applied on the spectrogram of the musical signal. The spectrogram is classicaly obtained as the power of the Fourier transform of small windows of the signal. Instead of using the Fourier transform, TL-NMF learns it from the signals.


Principle
=========

Given a matrix of positive entries V of size M x N, NMF learns matrices W and H of positive entries such that V is approximately WH.
W is of size M x K, and H of size K x N, where K is the rank of the factorization.
NMF is typically used in signal processing to factorize the spectrogram.
The signal is cut into N short windows of size M, to form a frames matrix Y of size M x N.
Then, a M x M DCT matrix, Phi, is used to form the spectrogram: V = (Phi Y)^2.

Instead of taking a fixed DCT transform, TL-NMF also learns Phi, straight from the signal.


Installation
============

To install the module `tlnmf`, you should first clone this repository::

    $ git clone https://github.com/pierreablin/tlnmf.git

And then install it using `pip`::

    $ pip install -e.

To check that everything worked, the following command should not return any error:

    $ python -c 'import tlnmf'



Use
===

The main function in `tlnmf` is `tlnmf.tl_nmf`. Given a frames matrix Y and a rank K, it outputs the learned Phi, W, H. Read the docstring of the function for an accurate description of its parameters.

The function `tlnmf.signal_to_frame` is there to build the frames matrix from the signal.

The file `examples/example_amstrong.py` contains a practical example on the song My Heart by Louis Amstrong.


Dependencies
============

    * numpy (>=1.8)
    * matplotlib (>=1.3)
    * scipy (>=0.19)
    * soundfile (>=0.10.2)
