#!/usr/bin/env python

import math
import sys
import time

import numpy as np
import pyopencl as cl

ftype = 'float16'
ftype_bytes = 2

itype = 'int32'
itype_bytes = 4

n_columns = 10000
column_size = 100


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
    __kernel void foo(
        __global const float *afferent_currents,
        __global const int *afferent_starts,
        __global const int *afferent_strides,
        __global const int *synapse_counts,
        __global const float *synapse_strengths,
        __global const int *synapse_starts,
        __global const int *synapse_strides,
        __global float *buf_rt,
        __global float *new_afferent_currents
                 )
    {
        int gid = get_global_id(0);

        int n_synapses = synapse_counts[gid];

        int aff_start = afferent_starts[gid];
        int aff_stride = afferent_strides[gid];

        int syn_start = synapse_starts[gid];
        int syn_stride = afferent_strides[gid];

        const __global float * aff = afferent_currents + aff_start;
        const __global float * syn = synapse_strengths + syn_start;

        float syn_contrib = 0.1f; // TODO: BIAS
        for (int i = 0; i < n_synapses; ++i)
        {
            syn_contrib += aff[i * aff_stride] * syn[i * syn_stride];
        }

          const float dt = 0.001;
          const float t_rc = 0.01;
          const float t_ref = 0.001;
          const float V_threshold = 1.0;

          float v = afferent_currents[gid];
          float rt = buf_rt[gid];

          float dV = dt / t_rc * (syn_contrib - v);
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

          //buf_spiked[gid] = spiked ? 1.0 : 0.0;
          buf_rt[gid] = new_rt;
          new_afferent_currents[gid] = new_voltage;
    }
    """).build()

foo = prg.foo
#foo.set_scalar_arg_dtypes((None, None, np.int32))

mf = cl.mem_flags

n_cells = n_columns * column_size

b_afferent_currents0 = cl.Buffer(ctx, mf.READ_WRITE, size=n_cells * ftype_bytes)
b_afferent_currents1 = cl.Buffer(ctx, mf.READ_WRITE, size=n_cells * ftype_bytes)
b_afferent_starts = cl.Buffer(ctx, mf.READ_ONLY, size=n_cells * itype_bytes)
b_afferent_strides = cl.Buffer(ctx, mf.READ_ONLY, size=n_cells * itype_bytes)

b_synapse_counts = cl.Buffer(ctx, mf.READ_ONLY, size=n_cells * itype_bytes)
b_synapse_strengths = cl.Buffer(ctx, mf.READ_ONLY,
        size=n_cells * column_size * ftype_bytes)
b_synapse_starts = cl.Buffer(ctx, mf.READ_ONLY, size=n_cells * itype_bytes)
b_synapse_strides = cl.Buffer(ctx, mf.READ_ONLY, size=n_cells * itype_bytes)

b_rt = cl.Buffer(ctx, mf.READ_WRITE,size=n_cells * ftype_bytes)


# -- initialize values in numpy

afferent_currents = np.zeros(n_cells, dtype=ftype)
afferent_starts = (
        np.zeros((n_columns, column_size), dtype=itype)
        + np.arange(n_columns, dtype=itype)[:, None]).flatten()
afferent_strides = np.ones(n_cells, dtype=itype)

synapse_counts = np.zeros(n_cells, dtype=itype) + column_size
synapse_strengths = np.arange(n_columns * column_size * column_size,
        dtype=itype)
synapse_strengths *= 37
synapse_strengths = (synapse_strengths % 73).astype(ftype) / 73.0
synapse_strengths.shape = (n_columns, column_size, column_size)
synapse_starts = np.arange(n_cells, dtype=itype)
synapse_strides = np.zeros(n_cells, dtype=itype) + n_cells

rval = np.empty(n_cells, dtype=ftype)

rt = np.zeros(n_cells, dtype=ftype)


# -- transfer values to device

cl.enqueue_copy(queue, b_afferent_currents0, afferent_currents, is_blocking=True)
cl.enqueue_copy(queue, b_afferent_currents1, afferent_currents, is_blocking=True)
cl.enqueue_copy(queue, b_afferent_starts, afferent_starts, is_blocking=True)
cl.enqueue_copy(queue, b_afferent_strides, afferent_strides, is_blocking=True)
cl.enqueue_copy(queue, b_synapse_counts, synapse_counts, is_blocking=True)
cl.enqueue_copy(queue, b_synapse_strengths, synapse_strengths, is_blocking=True)
cl.enqueue_copy(queue, b_synapse_starts, synapse_starts, is_blocking=True)
cl.enqueue_copy(queue, b_synapse_strides, synapse_strides, is_blocking=True)
cl.enqueue_copy(queue, b_rt, rt, is_blocking=True)

t0 = time.time()
n_iters = 25
for i in range(n_iters):
    for b_ac_in, b_ac_out in (
            (b_afferent_currents0, b_afferent_currents1),
            (b_afferent_currents1, b_afferent_currents0),
            ):
        event = foo(queue, (n_cells,), None,
                b_ac_in,
                b_afferent_starts,
                b_afferent_strides,
                b_synapse_counts,
                b_synapse_strengths,
                b_synapse_starts,
                b_synapse_strides,
                b_rt,
                b_ac_out)

event.wait()
dt = time.time() - t0

cl.enqueue_copy(queue, rval, b_afferent_currents0, is_blocking=True)
print rval
cl.enqueue_copy(queue, rval, b_afferent_currents1, is_blocking=True)
print rval
print '%fs per iteration' % (dt / (2 * n_iters))


