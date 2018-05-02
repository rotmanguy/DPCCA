from torch import diag,symeig,nn,save,sqrt,t,trace
from torch.autograd import Variable

def compute_mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    [D,V] = symeig(mat.data,eigenvectors=True)
    mat_pow = V.mm(diag((D.add(epsilon)).pow(pow_))).mm(t(V))
    mat_pow[mat_pow != mat_pow] = epsilon # For stability
    return Variable(mat_pow)

def activation_centered(Z, activ):
    if activ == 'sigmoid':
        activation = nn.Sigmoid()
    elif activ == 'tanh':
        activation = nn.Tanh()
    else:
        activation = nn.ReLU()
    Z = activation(Z) # Passing Z through same activation function of the main inputs
    Z = Z.sub(Z.mean(0).expand(Z.size())) # Removing mean from tensor
    return Z

def symmetric_average(mat):
    # Returning a symmetric matrix (mat = 0.5*(mat^T + mat)
    return t(mat).add(mat).div(2.0)

def compute_correlation(F, G):
    # Computing the canonical correlation of F and G
    split_size = F.size()[0]
    FG = (1.0/(split_size-1))*(t(F).mm(G))
    FF = (1.0/(split_size-1))*(t(F).mm(F))
    GG = (1.0/(split_size-1))*(t(G).mm(G))
    sqrt_trace_FF = sqrt(trace(FF))
    sqrt_trace_GG = sqrt(trace(GG))
    trace_FG = (trace(FG))
    correlation = trace_FG/(sqrt_trace_FF*sqrt_trace_GG)
    return correlation

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # Saving checkpoint
    print('saving checkpoint to file: ', filename)
    save(state, filename)

