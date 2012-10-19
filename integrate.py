D=1       # number of dimensions
N=100     # number of neurons
pstc=0.05 # time constant for filter (post-synaptic)
dt=0.001

from ccm.lib import nef

INPUT=[0.5]*D
input=nef.VectorNode(D)

# group of neurons representing A
a=nef.VectorNode(D)
a.configure(neurons=N)
a.configure_spikes(pstc=pstc,dt=dt)

# group of neurons representing B
b=nef.VectorNode(D)
b.configure(neurons=N)
b.configure_spikes(pstc=pstc,dt=dt)

input.connect(a)
a.connect(b,weight=pstc)
b.connect(b)


# extract all the information needed for the FPGA

data=open('integrate.csv','w')

# group A: only dependent on the INPUT state variable, only affect the A state variable
for i in range(N):
    bias=a.Jbias[i]
    encoder=[0.0]*(3*D)
    for j in range(D):
        encoder[j]=a.alpha[i]*a.basis[i][j]
    decoder=[0.0]*(3*D)    
    for j in range(D):
        decoder[j+D]=a.get_decoder()[i][j]        
    line='%d,%1.5g,%s,%s'%(i,bias,','.join(['%1.5g'%x for x in encoder]),','.join(['%1.5g'%x for x in decoder]))
    data.write(line+'\n')
    #print line

# group B: dependent on the A and B as follows: B=pstc*A+B
for i in range(N):
    bias=b.Jbias[i]
    encoder=[0.0]*(3*D)
    for j in range(D):
        encoder[j+D]=b.alpha[i]*b.basis[i][j]*pstc
    for j in range(D):
        encoder[j+2*D]=b.alpha[i]*b.basis[i][j]
    decoder=[0.0]*(3*D)    
    for j in range(D):
        decoder[j+2*D]=a.get_decoder()[i][j]        
    line='%d,%1.5g,%s,%s'%(i+100,bias,','.join(['%1.5g'%x for x in encoder]),','.join(['%1.5g'%x for x in decoder]))
    data.write(line+'\n')
    #print line

data.close()



# this is just to get access to a filtered value for A and B
#  -- the .value() function grabs the state variable *before* the filtering
a2=nef.VectorNode(D)
a.connect(a2)
b2=nef.VectorNode(D)
b.connect(b2)


input.set(INPUT)   # set the input value for A


for i in range(1000):
    a.tick(dt=dt)
    
    print i,a2.value(),b2.value()
    
    
    # various values for comparison:
    #   membrane voltage of neurons:  a.voltage, b.voltage
    #   whether or not each neuron spiked in the last time step:  a.spikes, b.spikes
    #   unfiltered value of B:  b.value()
    #   filtered value of B:    b2.value()

    # tau_rc=0.02  (this rarely ever changes -- hard-code it?)
    # tau_ref=0.002 (this rarely ever changes -- hard-code it?)



