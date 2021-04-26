# ==============================================================
# File name: Assignment5_template.py
# Author: NuAI Lab
# Date created: 2021-03-31
# Date last modified: 2021-03-31
# Python Version: 3.*
# Description: Code framework for eRBP
#
# Preliminaries:
# - install python 3.{5,6,7,8}
# - install pip: https://www.makeuseof.com/tag/install-pip-for-python/
# - install pip packages:
#   - pip install -U requests numpy matplotlib python_mnist
#
# Helpful resources:
# - matplotlib user's guide: https://matplotlib.org/users/index.html
# - NumPy user guide: https://docs.scipy.org/doc/numpy/user/
# =====================================================================
# For py2/3 compat/cleanliness:
# from __future__ import print_function, absolute_import, division
# ==============================================================

# =======================================
# START ASSIGNMENT 5 HERE
# -----------------------
# A "TODO" indicates a place for you to
# add or modify code.
# =======================================

## Imports
#
import numpy as np
import numpy.matlib
import mnist
import math

# Debug utilities
def nz(a):
    return np.count_nonzero(a) != 0
# Learning rule parameters
Imin = -0.05 # Minimum current for learning
Imax = 0.05 # Maximum current for learning
lr0 = 1e-4 # Learning rate in hidden layer
w_scale0 = 1e-3 # Weight scale in hidden layer
lr1 = 1e-5 # Learning rate at output layer
w_scale1 = 1e-3 # Weight scale at output layer
FPF = 0 # inhibits punishing target neuron (only use if training a specific output spike pattern)

# Neuron parameters
t_syn = 25
t_syn1 = 25 # Synaptic time-constant at output layer

t_m = 25 # Neuron time constant in hidden layer
t_mH = 10 # Neuron time constant in output layer
t_mU = 150 # Neuron time constant for error accumulation
t_mE = 150 # Error neuron time constant

R = t_m/10 # Membrane resistance in hidden layer
RH = t_mH/50 # Membrane resistance in output layer
RE = t_mE/150 # Membrane resistance in error neurons
RU = t_mU/150 # Membrane resistance of error compartment

Vth = 0.005 # Hidden neuron threshold
VthO = 0.005 # Output neuron threshold
VthE = 0.01 # Error neuron threshold

V_rest = 0 # Resting membrane potential
t_refr = 4 # Duration of refractory period

# Simulation parameters
tSim = 0.15 # Duration of simulation (seconds)
MaxF = 250 # maximum frequency of the input spikes
maxFL = 500 # maximum frequency of the target label spikes
dt = 1 # time resolution
dt_conv = 1e-3 # Data is sampled in ms
nBins = int(np.floor(tSim/dt_conv)) #total no. of time steps

# Network architecture parameters
n_h1 = 200 # no. of hidden neurons
dim = 28 # dim by dim is the dimension of the input images
n_in = dim*dim # no. of input neurons
n_out = 10 # no. of output neurons

# Generate forward pass weights
w_in = np.random.normal(0, w_scale0, (n_h1, n_in)) # input->hidden weights
w_out = np.random.normal(0, w_scale1, (n_out, n_h1)) # hidden->output weights

# Generate random feedback weights
w_err_h1p = np.random.uniform(-1, 1, (n_h1, n_out)) # false-pos-error->hidden weights
w_err_h1n = np.random.uniform(-1, 1, (n_h1, n_out)) # false-neg-error->hidden weights

# Load MNIST Data
n_train = 4000
n_test = 1000
maxE = 5 # no. of epochs

def make_spike_trains(freqs, n_steps):
    ''' Create an array of Poisson spike trains
        Parameters:
            freqs: Array of mean spiking frequencies.
            n_steps: Number of time steps
    '''
    r = np.random.rand(len(freqs), n_steps)
    spike_trains = np.where(r <= np.reshape(freqs, (len(freqs),1)), 1, 0)
    return spike_trains

def MNIST_to_Spikes(maxF, im, t_sim, dt):
    ''' Generate spike train array from MNIST image.
        Parameters:
            maxF: max frequency, corresponding to 1.0 pixel value
            FR: MNIST image (784,)
            t_sim: duration of sample presentation (seconds)
            dt: simulation time step (seconds)
    '''
    n_steps = np.int(t_sim / dt) #  sample presentation duration in sim steps
    freqs = im * maxF * dt # scale [0,1] pixel values to [0,maxF] and flatten
    SpikeMat = make_spike_trains(freqs, n_steps)
    return SpikeMat

from mnist.loader import MNIST
loader = MNIST('./mnist_dataset/') # replace with your MNIST path

# loader returns a list of 768-element lists of pixel values in [0,255]
# and a corresponding array of single-byte labels in [0-9]

TrainIm, TrainL = loader.load_training()
TrainIm = np.array(TrainIm) # convert to ndarray
TrainL = np.array(TrainL)
TrainIm = TrainIm / TrainIm.max() # scale to [0, 1] interval

TestIm, TestL = loader.load_testing()
TestIm = np.array(TestIm) # convert to ndarray
TestL = np.array(TestL)
TestIm = TestIm / TestIm.max() # scale to [0, 1] interval

# Randomly select train and test samples
trainInd = np.random.choice(len(TrainIm), n_train, replace=False)
TrainIm = TrainIm[trainInd]
TrainLabels = TrainL[trainInd]

testInd = np.random.choice(len(TestIm), n_test, replace=False)
TestIm = TestIm[testInd]
TestLabels = TestL[testInd]

# How often to print trace message to get max 100 traces/epoch
#
trace_intrvl = pow(10, math.ceil(math.log10(n_train/100)))

for e in range(maxE): # for each epoch
    for u in range(n_train): # for each training pattern
        if u % trace_intrvl == 0:
            print(f'Training epoch {e+1}/{maxE} pattern {u}/{n_train}')

        # Generate poisson data and labels
        spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u], tSim, dt_conv)
        fr_label = np.zeros(n_out)
        fr_label[TrainLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt_conv, nBins) # target spikes

        # Initialize hidden layer variables
        I1 = np.zeros(n_h1)
        V1 = np.zeros(n_h1)
        U1 = np.zeros(n_h1)

        # Initialize output layer variables
        I2 = np.zeros(n_out)
        V2 = np.zeros(n_out)
        U2 = np.zeros(n_out)

        # Initialize error neuron variables
        Verr1 = np.zeros(n_out)
        Verr2 = np.zeros(n_out)

        # Initialize firing time variables
        ts1 = np.full(n_h1, -t_refr)
        ts2 = np.full(n_out, -t_refr)
        tsE1 = np.full(n_out, -t_refr)
        tsE2 = np.full(n_out, -t_refr)

        SE1T = np.zeros((10, nBins)) # to record error neuron spiking
        SE2T = np.zeros((10, nBins))

        for t in range(nBins):
            # Forward pass
            
            # Find input neurons that spike
            fired_in = np.nonzero(spikeMat[:, t])

            # Update synaptic current into hidden layer
            I1 += (dt/t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

            # Update hidden layer membrane potentials
            V1 += (dt/t_m) * ((V_rest - V1) + I1 * R)
            V1[V1 < -Vth/10] = -Vth/10 # Limit negative potential

            # If neuron in refractory period, prevent changes to membrane potential
            refr1 = (t*dt - ts1 <= t_refr)
            V1[refr1] = 0

            fired = np.nonzero(V1 >= Vth) # Hidden neurons that spiked
            V1[fired] = 0 # Reset their membrane potential to zero
            ts1[fired] = t # Update their most recent spike times

            ST1 = np.zeros(n_h1) # Hidden layer spiking activity
            ST1[fired] = 1 # Set neurons that spiked to 1

            # Repeat the process for the output layer
            I2 += (dt/t_syn1)*(w_out.dot(ST1) - I2)

            V2 += (dt/t_mH)*((V_rest - V2) + I2*(RH))
            V2[V2 < -VthO/10] = -VthO/10

            refr2 = (t*dt - ts2 <= t_refr)
            V2[refr2] = 0
            fired2 = np.nonzero(V2 >= VthO)

            #if len(fired2[0]) > 0:
            #   print(f'{len(fired2[0])} output neurons fired')


            V2[fired2] = 0
            ts2[fired2] = t

            #-----------------------------------------------------------------
            # TODO: Compute error (used as input to error neurons)
            #-----------------------------------------------------------------

            # Answer:
            # Make array of output neuron spikes
            ST2 = np.zeros(n_out)
            ST2[fired2] = 1

            # Compare with target spikes for this time step
            Ierr = (ST2 - s_label[:, t])

            #-----------------------------------------------------------------
            # TODO: Update error neurons and check for firing (positive and
            # negative error neurons)
            # If an error neuron fires, it spikes and has it membrane
            # potential decreased by VthE
            #-----------------------------------------------------------------

            # Answer:
            # Update false-positive error neuron membrane potentials
            Verr1 += (dt/t_mE)*(Ierr*RE)
            Verr1[Verr1 < -VthE/10] = -VthE/10 # Limit negative potential to -VthE/10

            #-----------------------------------------------------------------
            # TODO: Update second compartment (U1/U2) with error feedback
            #-----------------------------------------------------------------

            # Answer:
            ## Process spikes in false-positive error neurons
            fired_err1 = np.nonzero(Verr1 >= VthE)
            Verr1[fired_err1] -= VthE

            # Don't penalize "false positive" spikes on the target
            Verr1[TrainLabels[u]] *= FPF

            tsE1[fired_err1] = t # update most recent spike times

            # Make array of false-positive error neuron spikes
            Serr1 = np.zeros(n_out)
            Serr1[fired_err1] = 1
            SE1T[:,t] = Serr1

            # Update false-negative error neuron membrane potentials
            Verr2 -= (dt/t_mE)*(Ierr*RE)
            Verr2[Verr2 < -VthE/10] = -VthE/10

            ## Process spikes in false-negative error neurons
            fired_err2 = np.nonzero(Verr2 >= VthE)
            Verr2[fired_err2] -= VthE
            tsE2[fired_err2] = t
            #print(f'fired_err1:{len(fired_err1[0])}')
            #print(f'fired_err2:{len(fired_err2[0])}')

            # Make array of false-negative error neuron spikes
            Serr2 = np.zeros(n_out)
            Serr2[fired_err2] = 1
            SE2T[:,t] = Serr2

            # Update hidden neuron error compartments (using random weights)
            U1 += (dt/t_mU)*(-U1 + (w_err_h1p.dot(Serr1) - w_err_h1n.dot(Serr2))*RU)

            # Update output neuron error compartments
            U2 += (dt/t_mU)*(-U2 + (Serr1 - Serr2)*RU)

            #-----------------------------------------------------------------
            # TODO: Compute hidden layer weight updates
            # Check if input neurons fired
            # Check if hidden neuron current (I1) is between Imin and Imax
            # Update weights (w_in) based on learning rate (lr0) and error (U1)
            #-----------------------------------------------------------------

            # Answer:
            if len(fired_in[0]) != 0:
                pre_ind = fired_in
                post_ind = np.nonzero((I1>Imin) & (I1<Imax))

                UF = U1[post_ind[0]] # np.nozero wraps the array in a tuple

                dw = -lr0*np.matlib.repmat(np.reshape(UF, (len(UF),1)), 1, len(pre_ind[0]))
                w_in[np.ix_(post_ind[0], pre_ind[0])] += dw

            #-----------------------------------------------------------------
            # TODO: Compute output layer weight updates
            # Check if hidden neurons fired
            # Check if output neuron current (I2) is between Imin and Imax
            # Update weights (w_out) based on learning rate (lr1) and error (U2)
            #-----------------------------------------------------------------

            # Answer:
            if len(fired[0]) != 0:
                pre_ind = fired
                post_ind = np.nonzero((I2>Imin) & (I2<Imax))
                UF = U2[post_ind[0]]

                dw = -lr1*np.matlib.repmat(np.reshape(UF, (len(UF),1)), 1, len(pre_ind[0]))

                w_out[np.ix_(post_ind[0], pre_ind[0])] +=  dw

    # TODO: Check train and test accuracy here.
    # If the output neuron with highest firing rate matches the target
    # neuron, and that rate is > 0, then the sample was classified correctly

    # Answer:
    def check_accuracy(images, labels):
        """Present a set of labeled images to the network and count correct inferences
        :param images: images
        :param labels: labels
        :return: fraction of labels correctly inferred
        """
        numCorrect = 0

        for u in range(len(images)):
            cnt = np.zeros(n_out)
            spikeMat = MNIST_to_Spikes(MaxF, images[u], tSim, dt_conv)

            # Initialize hidden layer variables
            I1 = np.zeros(n_h1)
            V1 = np.zeros(n_h1)

            # Initialize output layer variables
            I2 = np.zeros(n_out)
            V2 = np.zeros(n_out)

            # Initialize firing time variables
            ts1 = np.full(n_h1, -t_refr)
            ts2 = np.full(n_out, -t_refr)
            tsE1 = np.full(n_out, -t_refr)
            tsE2 = np.full(n_out, -t_refr)

            for t in range(nBins):
                # Update hidden neuron synaptic currents
                I1 += (dt/t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

                # Update hidden neuron membrane potentials
                V1 += (dt/t_m) * ((V_rest - V1) + I1 * R)
                V1[V1 < -Vth/10] = -Vth/10 # Limit negative potential to -Vth/10

                # Clear membrane potential of hidden neurons that spiked more
                # recently than t_refr
                V1[t*dt - ts1 <= t_refr] = 0

                ## Process hidden neuron spikes
                fired = np.nonzero(V1 >= Vth) # Hidden neurons that spiked
                V1[fired] = 0 # Reset their membrane potential to zero
                ts1[fired] = t # Update their most recent spike times

                # Make array of hidden-neuron spikes
                ST1 = np.zeros(n_h1)
                ST1[fired] = 1

                # Update output neuron synaptic currents
                I2 += (dt/t_syn1)*(w_out.dot(ST1) - I2)

                # Update output neuron membrane potentials
                V2 += (dt/t_mH)*((V_rest - V2) + I2*(RH))
                V2[V2 < -VthO/10] = -VthO/10 # Limit negative potential to -Vth0/10

                # Clear V of output neurons that spiked more recently than t_refr
                refr2 = (t*dt - ts2 <= t_refr)
                V2[refr2] = 0

                ## Process output spikes
                fired2 = np.nonzero(V2 >= VthO) # output neurons that spikes
                V2[fired2] = 0 # Reset their membrane potential to zero
                ts2[fired2] = t # Update their most recent spike times

                # Make array of output neuron spikes
                ST2 = np.zeros(n_out)
                ST2[fired2] = 1

                cnt += ST2

            if np.count_nonzero(cnt) != 0:  # Avoid counting no spikes as predicting label 0
                prediction = np.argmax(cnt)
                target = labels[u]

                if prediction == target:
                    numCorrect += 1

        return numCorrect/len(images)


    train_acc = check_accuracy(TrainIm, TrainLabels)
    test_acc = check_accuracy(TestIm, TestLabels)
    print(f'Epoch {e} train_acc = {train_acc}  test_acc = {test_acc}')
