#!/usr/bin/env python

import numpy
import time
import math
import sys

if 0:
    import pyopencl as cl
    import numpy
    import numpy.linalg as la

    a = numpy.random.rand(50000).astype(numpy.float32)
    b = numpy.random.rand(50000).astype(numpy.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

    prg = cl.Program(ctx, """
        __kernel void sum(__global const float *a,
        __global const float *b, __global float *c)
        {
          int gid = get_global_id(0);
          c[gid] = a[gid] + b[gid];
        }
        """).build()

    prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)

    a_plus_b = numpy.empty_like(a)
    cl.enqueue_copy(queue, a_plus_b, dest_buf)

    print la.norm(a_plus_b - (a+b))


class FPGA:
    def __init__(self,data):
        self.N=len(data)
        self.D=(len(data[0])-2)/2
        print self.N, self.D

        data=numpy.array(data)
        self.encoders=data[:,2:(2+self.D)]
        self.decoders=data[:,(2+self.D):]

        self.state = numpy.zeros(self.D)
        self.voltage = numpy.zeros(self.N)
        self.refractory_time = numpy.zeros(self.N)
        self.Jm_prev = numpy.zeros(self.N)
        self.J_bias = data[:,1].copy()

        self.t_rc = 0.01
        self.t_ref = 0.001
        self.dt=0.001
        self.pstc=0.05

    def tick(self):
        dt = self.dt
        v = self.voltage
        rt = self.refractory_time

        Jm = numpy.dot(self.encoders, self.state) + self.J_bias

        # Euler's method
        dV = dt / self.t_rc * (self.Jm_prev - v)

        v += dV
        v = numpy.maximum(v, 0)

        # do accurate timing for when the refractory period ends   (probably not needed for FPGA neuron model)
        post_ref = 1.0 - (rt - dt) / dt
        v *= numpy.clip(post_ref, 0, 1)

        V_threshold = 1
        spiked = v > V_threshold
        overshoot=(v - V_threshold) / dV
        spiketime = dt * (1.0 - overshoot)

        new_state = numpy.zeros(self.D)

        new_refractory_time = numpy.where(spiked,
                spiketime + self.t_ref,
                rt - dt)

        new_state=numpy.dot(spiked, self.decoders) / dt
        new_voltage = v * (1 - spiked)

        # apply the filter
        decay=math.exp(-dt/self.pstc)
        self.state=self.state*decay+(1-decay)*new_state

        # update other state variables
        self.voltage = new_voltage
        self.refractory_time = new_refractory_time
        self.Jm_prev = Jm


def read_data(filename):
    for line in open(filename).readlines():
        yield [float(x) for x in line.strip().split(',')]

f = FPGA(list(read_data(sys.argv[1])))

n_iters = 1000
t0 = time.time()
for j in range(n_iters):
    f.state[:f.D/3]=0.5
    f.tick()
print f.state
print 'time per iteration', (time.time() - t0) / n_iters

