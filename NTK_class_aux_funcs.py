from sklearn.preprocessing import normalize
import math
import numpy as np
import random
np.random.seed()

def create_data(n,d):
    """
    INPUT: 1)n: number of input vector and output values to sample
           2)d: dimension of input vectors
    OUTPUT: Returns n random vectors of dimension d ~N(0,1_d), normalised to the unit sphere, and n random numbers ~ N(0,1)
    OUTPUT SHAPE: ([[x1],...,[xn]],[y1,...,yn])
    """

    X = normalize(np.random.normal(size=(n,d)), axis=1, norm='l2')
    Y = np.random.uniform(low=-1,high=1,size=n)
    return(X,Y)

def gen_norm_excl_inter(low=-1,high=1,size=1):
    """
    INPUT: 1) low, high: real numbers, low < high
           2) size: number of entries to generate
    OUPTUT: Returns a vector with m entries, each generated as standard normal excluding the interval (low,high)
    """
    res = []                                #to store generated values
    for i in range(size):
        val = np.random.normal()            #generate a first value
        while val > low and val < high:     #while in interval to exclude keep generating again until it isnt
            val = np.random.normal()
        res.append(val)                     #store value and repeat
    return(res)

def create_weights(m,d,rademacher = True):
    """
    INPUT: m,d real numbers, rademacher a boolean that determines wether the a_r are initialized as Rademacher or Standard Normal excluding (-1,1)
    OUTPUT: W a matrix of size d*m with iid ~N(0,1) entries, a a vector of length m with entries ~ Rademacher/Normal
    OUTPUT SHAPE: ([a1,...,am],[[w1],...,[wm]])
    """
    if rademacher:                          
        a = np.random.choice(np.array([int(-1),int(1)]),size = m)
    else:
        a = gen_norm_excl_inter(low=-1,high=1,size=m)
    W = np.random.normal(size=(m,d))
    return(a,W)

def create_weights(m,d,rademacher = True):
    """
    INPUT: m,d real numbers, rademacher a boolean that determines wether the a_r are initialized as Rademacher or Standard Normal excluding (-1,1)
    OUTPUT: W a matrix of size d*m with iid ~N(0,1) entries, a a vector of length m with entries ~ Rademacher/Normal
    OUTPUT SHAPE: ([a1,...,am],[[w1],...,[wm]])
    """
    if rademacher:                          
        a = np.random.choice(np.array([int(-1),int(1)]),size = m)
    else:
        a = gen_norm_excl_inter(low=-1,high=1,size=m)
    W = np.random.normal(size=(m,d))
    return(a,W)

#--------------Surrogate heaviside--------------

def sigma_beta(x,b):
    """
    INPUT: 1)x: value in R
           2)b: beta, the parameter for the surrogate heaviside function
    OUTPUT: Value of the surrogate heaviside with parameter beta at x
    """
    if abs(x) < b:
        return(0.5 + x/(2*b))
    elif x >= b:
        return(1)
    else:
        return(0)

#----------------------------

#--------------Surrogate heaviside derivative--------------

def sigma_beta_der(x,b):
    """
    INPUT: 1)x: value in R
           2)b: beta, the parameter for the surrogate heaviside function
    OUTPUT: Value of the gradient of the surrogate with parameter beta at x
    """
    if abs(x) <= b:
        return 1/(2*b)
    else:
        return 0
    
#----------------------------

#--------------Loss function--------------
    
def compute_loss(X,y):
    """
    INPUT: 1)X: Vector of dimension n containing the outputs of the neural network
           2)y: Vector of dimension n containing the desired output
    OUTPUT: Squared loss
    """
    return(0.5*np.sum([(X[i]-y[i])**2 for i in range(len(y))]))

#----------------------------

#--------------Compute output using real heaviside--------------

def NTK_output(a,W,X):
    """
    INPUT: 1)a: outer weights
           2)W: inner weights
           3)X: input data
    OUTPUT: Output of shallow heaviside neural network, for each input data
    """
    return([np.sum([a[r] for r in range(len(W)) if np.dot(W[r],X[j]) >= 0])/(math.sqrt(len(W))) for j in range(len(X))])

#----------------------------

#--------------Compute output using approximate heaviside--------------

def NTK_output_beta(a,W,X,b):
    """
    INPUT: 1)a: outer weights
           2)W: inner weights
           3)X: input data
           4)b: parameter for surrogate activation function
    OUTPUT: Output of shallow approximate heaviside neural network, for each input data
    """
    return([np.sum([a[r]*sigma_beta(np.dot(W[r],X[j]),b) for r in range(len(W))])/(math.sqrt(len(W))) for j in range(len(X))])

#----------------------------

class NTK_shallow:
    
    #--------------Initialisation--------------

    def __init__(self,n,d,m,b,train_a = True,X=[],y=[],a=[],W=[]):
        #store given variables
        self.n = n                  #number of input vectors
        self.d = d                  #dimension of input vectors
        self.m = m                  #number of neurons
        self.b = b                  #parameter for surrogate heaviside
        self.train_a = train_a      #if true, the outer weights are trained

        #generate random data if required and store it
        if len(X) == 0:
            if len(y) == 0:
                self.X,self.y = create_data(self.n,self.d)
            else:
                self.X = create_data(self.n,self.d)[0]
                self.y = np.copy(y)
        else:
            if len(y) == 0:
                self.y = create_data(self.n,self.d)[1]
                self.X = np.copy(X)
            else:
                self.X = np.copy(X)
                self.y = np.copy(y)

        #initialise weights of the network if required and store them
        if len(a) == 0:
            if len(W) == 0:
                self.a, self.W = create_weights(self.m,self.d)
            else:
                self.a = create_weights(self.m,self.d)[0]
                self.W = np.copy(W)
        else:
            if len(W) == 0:
                self.W = create_weights(self.m,self.d)[1]
                self.a = np.copy(a)
            else:
                self.W = np.copy(W)
                self.a = np.copy(a)

        #compute current value of f and store it
        self.f = NTK_output(self.a,self.W,self.X)

        #create object to store evolution of inner weights over time
        self.W_iters = [np.copy(self.W)]

        #compute initial loss and store it
        self.losses = [compute_loss(self.f,self.y)]

        #create object to store evolution of outer weights over time
        self.a_iters = [np.copy(self.a)]

        #We want to study if the outer weights or inner weights contribute more to lowering the loss at early vs late stages of training
        self.loss_diffs = []

    #----------------------------

    #--------------Gradient wrt inner weights--------------
        
    def grad_w(self,r):
        """
        INPUT: r: index of inner weight to compute gradient for
        OUTPUT: gradient wrt w_r
        """
        return(np.sum([(self.f[j] - self.y[j]) * self.a[r] * sigma_beta_der(np.dot(self.W[r],self.X[j]),self.b) * self.X[j] for j in range(self.n)],axis=0)/(math.sqrt(self.m)))
    
    #----------------------------

    #--------------Gradient wrt outer weights--------------
    
    def grad_a(self,r):
        """
        INPUT: r: index of outer weight to compute gradient for
        OUTPUT: gradient wrt a_r
        """
        return(np.sum([(self.f[j] - self.y[j]) for j in range(self.n) if np.dot(self.W[r],self.X[j]) >= 0])/(math.sqrt(self.m)))
    
    #----------------------------

    #--------------Update weights of the network--------------

    def update_weights(self,t):
        """
        INPUT: t: size of step in time per iteration
        OUTPUT: No output, updates the weights by substracting t times the corresponding gradient
        Also updates the value of f, store the loss in self.losses, and if the outer weights are trained, stores the loss difference generated
        by the inner and outer weights respectively
        """
        if self.train_a:
            self.a = np.copy([self.a[r]-self.grad_a(r)*t for r in range(self.m)])   #update outer weights, copy as a is a reference type variable
        
        for r in range(self.m):
            self.W[r] = np.copy(self.W[r] - self.grad_w(r)*t)   #update inner weights, copy as W is a reference type variable
        
        self.W_iters.append(np.copy(self.W))                    #store evolution of inner weights
        self.a_iters.append(np.copy(self.a))                    #store evolution of outer weights
        self.f = NTK_output(self.a,self.W,self.X)               #update value of f
        self.losses.append(compute_loss(self.f,self.y))         #compute current loss and store it
        if self.train_a:                                        #compute loss difference generated by w and a respectively and store it
            self.loss_diffs.append([compute_loss(NTK_output(self.a,self.W_iters[-2],self.X),self.y)-self.losses[-2],compute_loss(NTK_output(self.a_iters[-2],self.W,self.X),self.y)-self.losses[-2]])

    #----------------------------

    #--------------Simulate num steps of lenght t--------------

    def run_sim(self,t,num):
        """
        INPUT: 1)t: size of time step per iteration
               2)num: number of steps for the simulation
        OUTPUT: No output, runs the simulation of the given parameters 
        """
        for i in range(num):
            self.update_weights(t)
            
    #----------------------------

    #--------------Compute different quantities that may be of interest--------------

    def comp_wr_x_ar_iters(self,i):
        """
            INPUT: i: index of chosen input data point
            OUTPUT: <w_r(t),x_i>/a_r(t)
        """
        return [[np.dot(self.W_iters[t][r],self.X[i])/self.a_iters[t][r] for r in range(self.m)] for t in range(len(self.W_iters))]
    
    def comp_output_iters_beta(self):
        """
            INPUT: /
            OUTPUT: Output vectors with approximate heaviside over time
        """
        return [NTK_output_beta(a=self.a_iters[t],W=self.W_iters[t],X=self.X,b=self.b) for t in range(len(self.W_iters))]
    
    def comp_output_iters(self):
        """
            INPUT: /
            OUTPUT: Output vectors with true heaviside over time
        """
        return [NTK_output(a=self.a_iters[t],W=self.W_iters[t],X=self.X) for t in range(len(self.W_iters))]

    #----------------------------