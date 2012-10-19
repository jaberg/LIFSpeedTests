#!/usr/bin/env python

import numpy
import math
import sys


class FPGA:
    def __init__(self,data):
        self.N=len(data)
        self.D=(len(data[0])-2)/2
        print self.N,self.D
        
        data=numpy.array(data)
        self.encoders=data[:,2:(2+self.D)]
        self.decoders=data[:,(2+self.D):]
        
        self.state=numpy.zeros(self.D)
        self.voltage=numpy.zeros(self.N)
        self.refractory_time=numpy.zeros(self.N)
        self.J_bias=data[:,1]
        self.Jm_prev=None
        self.t_rc=0.01
        self.t_ref=0.001
        
        self.dt=0.001
        self.pstc=0.05
        
    def tick(self):
        Jm=numpy.dot(self.encoders,self.state)+self.J_bias
        dt=self.dt
        if self.Jm_prev is None: self.Jm_prev=Jm
        v=self.voltage

        # Euler's method
        dV=dt/self.t_rc*(self.Jm_prev-v)

        self.Jm_prev=Jm
        v+=dV
        v=numpy.maximum(v,0)


        # do accurate timing for when the refractory period ends   (probably not needed for FPGA neuron model)
        self.refractory_time-=dt
        post_ref=1.0-self.refractory_time/dt  
        v=numpy.where(post_ref>=1,v,v*post_ref) # scale by amount of time outside of refractory period
        v=numpy.where(post_ref<=0,0,v)  # set to zero during refactory period
        
        new_state=numpy.zeros(self.D)        
        V_threshold=1
        for ii in numpy.where(v>V_threshold):
          for i in ii:
            overshoot=(v[i]-V_threshold)/dV[i]
            spiketime=dt*(1.0-overshoot)
            self.refractory_time[i]=spiketime+self.t_ref
            new_state+=self.decoders[i]/dt                 # it's probably a good idea to pre-multiply decoders by dt to avoid this computation
            v[i]=0
        self.voltage=v
        
        # apply the filter
        decay=math.exp(-dt/self.pstc)
        self.state=self.state*decay+(1-decay)*new_state
        
def read_data(filename):
    for line in open(filename).readlines():
        yield [float(x) for x in line.strip().split(',')]            

f=FPGA(list(read_data(sys.argv[1])))
for j in range(1000):
    f.state[:f.D/3]=0.5
    f.tick()
    print f.state
