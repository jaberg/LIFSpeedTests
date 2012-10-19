#!/usr/bin/env python

import numpy
import math
import sys
import time

import pyopencl as cl

dtype = 'float32'
dtype_bytes = 4

class FPGA:
    def __init__(self, data):
        self.N=len(data)
        self.D=(len(data[0])-2)/2
        print self.N, self.D

        self.t_rc = 0.01
        self.t_ref = 0.001
        self.dt = 0.001
        self.pstc = 0.05
        self.V_threshold = 1.0

        data = numpy.asarray(data, dtype=dtype)
        self.encoders = data[:,2:(2+self.D)].copy()
        self.decoders = data[:,(2+self.D):].copy()

        self.state = numpy.zeros(self.D).astype(dtype)
        self.J_bias = data[:,1].copy()

        ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        self.buf_Jm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=numpy.zeros(self.N).astype(dtype))

        self.buf_Jm_prev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=numpy.zeros(self.N).astype(dtype))

        self.buf_v = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=numpy.zeros(self.N).astype(dtype))

        self.buf_rt = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                hostbuf=numpy.zeros(self.N).astype(dtype))

        self.buf_spiked = cl.Buffer(ctx, mf.WRITE_ONLY, self.N * dtype_bytes)

        prg = cl.Program(ctx, """
            __kernel void foo(
                __global const float *buf_Jm,
                __global float *buf_Jm_prev,
                __global float *buf_v,
                __global float *buf_rt,
                __global float *buf_spiked)
            {
              int gid = get_global_id(0);

              const float dt = %(dt)s;
              const float t_rc = %(t_rc)s;
              const float t_ref = %(t_ref)s;
              const float V_threshold = %(V_threshold)s;

              float v = buf_v[gid];
              float Jm = buf_Jm[gid];
              float Jm_prev = buf_Jm_prev[gid];
              float rt = buf_rt[gid];

              float dV = dt / t_rc * (Jm_prev - v);
              v += dV;
              float post_ref = - rt / dt;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0 : post_ref < 1 ? post_ref : 1)
                  : 0;
              int spiked = v > V_threshold;
              float overshoot = (v - V_threshold) / dV;
              float spiketime = dt * (1.0 - overshoot);

              float new_voltage = v * (1.0 - spiked);
              float new_rt = spiked ? spiketime + t_ref : rt - dt;

              buf_Jm_prev[gid] = Jm;
              buf_spiked[gid] = spiked ? 1.0 : 0.0;
              buf_rt[gid] = new_rt;
              buf_v[gid] = new_voltage;
            }
            """ % self.__dict__).build()

        self.prg = prg


    @property
    def spiked(self):
        rval = numpy.empty(self.N, dtype=dtype)
        cl.enqueue_copy(self.queue, rval, self.buf_spiked)
        return rval

    @profile
    def tick(self):

        Jm = numpy.dot(self.encoders, self.state) + self.J_bias
        #print 'state', self.state
        #print 'Jm', Jm
        #cl.enqueue_copy(self.queue, self.buf_Jm, Jm, is_blocking=False)
        self.prg.foo(self.queue, (self.N,), None,
            self.buf_Jm, self.buf_Jm_prev,
            self.buf_v, self.buf_rt, self.buf_spiked)

        #spiked = self.spiked
        spiked = numpy.zeros(self.N, dtype=dtype)

        if 0:
            cl.enqueue_copy(self.queue, self.Jm_prev, self.buf_Jm_prev)
            print 'Jm_prev', self.Jm_prev[:10]
        if 0:
            voltage = numpy.empty(self.N, dtype=dtype)
            cl.enqueue_copy(self.queue, voltage, self.buf_v)
            print 'voltage', voltage[10:]

            #cl.enqueue_copy(self.queue, self.refractory_time, self.buf_rt)
        #print self.spiked

        new_state = numpy.dot(spiked, self.decoders) / self.dt

        # apply the filter
        decay = math.exp(-self.dt / self.pstc)
        self.state *= decay
        self.state += ( 1 - decay) * new_state


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

