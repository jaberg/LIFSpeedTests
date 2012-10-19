# RUN LIKE
# THEANO_FLAGS=floatX=float32

import theano
from theano import shared, function
import numpy as np
from theano import tensor as TT

floatX= theano.config.floatX
sharedX = lambda x: shared(x.astype(floatX))

import numpy
import math
import sys


class FPGA2:
    def __init__(self,data):
        self.N=len(data)
        self.D=(len(data[0])-2)/2
        print self.N,self.D
        
        data=numpy.array(data).astype(floatX)
        self.encoders=data[:,2:(2+self.D)]
        self.decoders=data[:,(2+self.D):]
        
        self.state=sharedX(numpy.zeros(self.D))
        self.voltage=sharedX(numpy.zeros(self.N))
        self.refractory_time=sharedX(numpy.zeros(self.N))
        self.J_bias=sharedX(data[:,1])
        self.Jm_prev=sharedX(np.zeros_like(data[:,1]))
        self.t_rc=0.01
        self.t_ref=0.001
        
        self.dt=0.001
        self.pstc=0.05

        state = TT.set_subtensor(self.state[0], .5)

        Jm=TT.dot(self.encoders, state)+self.J_bias
        dt = self.dt
        v = self.voltage
        # Euler's method
        dV=dt / self.t_rc * (self.Jm_prev - v)
        v += dV
        v = TT.maximum(v, 0)

        post_ref=1.0 - (self.refractory_time - dt)/dt  


        # do accurate timing for when the refractory period ends   (probably not needed for FPGA neuron model)
        v *= TT.clip(post_ref, 0, 1)

        V_threshold=1

        spiked = TT.switch(v > V_threshold, 1.0, 0.0)

        # -- refractory stuff
        overshoot=(v-V_threshold)/dV
        spiketime=dt*(1.0-overshoot)
        new_refractory_time = TT.switch(
            spiked,
            spiketime + self.t_ref,
            self.refractory_time - dt)

        new_v = v
        
        # apply the filter
        decay=math.exp(-dt/self.pstc)
        new_state= (state*decay
                    + (1-decay) * TT.dot(spiked, self.decoders)/dt)

        self.tick = function([], [],
                     updates={
                         self.voltage: new_v * (1 - spiked),
                         self.state: new_state,
                         self.refractory_time: new_refractory_time,
                         self.Jm_prev: Jm,
                     })


        
def read_data(filename):
    for line in open(filename).readlines():
        yield [float(x) for x in line.strip().split(',')]            

f=FPGA2(list(read_data(sys.argv[1])))
import time
theano.printing.debugprint(f.tick)
n_iters = 1000
print 'Running %i iterations' % n_iters
t0 = time.time()
f.tick.fn(n_calls=n_iters)
#for j in range(10000):
print 'Time per iteration', ((time.time() - t0) / n_iters)
print 'Final state vector', f.state.get_value()

