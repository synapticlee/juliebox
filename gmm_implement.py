#!/usr/bin/env python
# coding: utf-8

# ### Implementation of gaussian mixture model (aka mixture of gaussians) 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import multivariate_normal as mvn


# In[2]:


NClust = 3;


# In[4]:


dataFname = 'foo.npy'
X = np.load(dataFname)


# In[5]:


plt.scatter(X[:,0], X[:,1])
plt.axis('square');


# In[6]:


'''kmeans to have initial means'''
def kmeans(X,NClust,NIters=10):
    i = np.random.permutation(X.shape[0])[:NClust]
    centers = X[i]
    for itr in range(1,NIters):
        labels = pairwise_distances_argmin(X,centers)
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(NClust)])
        centers = new_centers
    return centers,labels


# In[25]:


def initParams(X,NClust):
    NDim            = X.shape[1]
    mu              = np.zeros((NClust, NDim))
    cov             = np.zeros((NClust, NDim, NDim))
    centers,labels  = kmeans(X,NClust)
    NEach           = np.bincount(labels)
    pi              = NEach/X.shape[0] #mixing proportions
    assert np.isclose(np.sum(pi),1), 'mixing proportions do not sum to 1'
    for lb in np.unique(labels):
        ix          = np.where(labels==lb)
        mu[lb,:]    = np.mean(X[ix,:])
        cov[lb,:,:] = np.dot(pi[lb] * np.squeeze(np.transpose(X[ix,:] - mu[lb,:])), 
                             np.squeeze(X[ix,:] - mu[lb,:])) / NEach[lb]
    return mu,cov,pi


# In[9]:


'''compute posterior probability that each datapoint is in each cluster, 
i.e. the responsibilities'''
def eStep(X,mu,cov,pi,NClust):
    NCells = X.shape[0] 
    posterior = np.zeros((NCells,NClust))
    for cl in range(NClust):
        likelihood = mvn.pdf(X, mu[cl,:],cov[cl,:,:])
        prior = pi[cl]
        posterior[:,cl] = likelihood*prior 
    respb = posterior 
    resb_norm = np.sum(respb, axis=1)[:,np.newaxis]
    gamma = respb / resb_norm
    return gamma


# In[10]:


gamma = eStep(X,mu,cov,pi,NClust)


# In[17]:


def mStep(X,gamma,NClust):
    NDim = X.shape[1]
    newPi = np.mean(gamma, axis=0) #mean per cluster
    newCov = np.zeros((NClust,NDim,NDim))
    '''new centers is "weighted (by responsibility) average", returns NDim x NClust'''
    newMu = np.transpose(1/np.sum(gamma,axis=0) * np.dot(gamma.T, X).T)
    for cl in range(NClust):
        meanSub = X - newMu[cl,:]
        gammaDiag = np.matrix(np.diag(gamma[:,cl]))
        covRaw = meanSub.T * gammaDiag * meanSub
        newCov[cl,:,:] = 1/np.sum(gamma,axis=0)[cl] * covRaw 
    return newPi,newMu,newCov


# In[18]:


pi,mu,cov = mStep(X,gamma,NClust)


# In[19]:


def getLoss(X,pi,mu,cov,NClust):
    NData = X.shape[0]
    loss = np.zeros((NData,NClust))
    for cl in range(NClust):
        dist = mvn(mu[cl,:], cov[cl],allow_singular=True)
        currloss = gamma[:,cl] * (np.log(pi[cl]+0.00001)+
                                    dist.logpdf(X)-np.log(gamma[:,cl]+0.000001))
        loss[:,cl] = currloss
    finalLoss = np.sum(loss)
    return finalLoss


# In[28]:


def fit(X,mu,pi,cov,NClust,NIters):
    for run in range(NIters):  
        gamma  = eStep(X,mu,cov,pi,NClust)
        pi, mu, cov = mStep(X,gamma,NClust)
        loss = getLoss(X, pi, mu, cov,NClust)
    return pi,mu,cov

def predict(X,mu,pi,cov,NClust):
    labels = np.zeros((X.shape[0], NClust))

    for cl in range(NClust):
        labels [:,cl] = pi[cl] * mvn.pdf(X, mu[cl,:], cov[cl])
    labels  = labels .argmax(1)
    return labels 


# In[26]:



mu,cov,pi = initParams(X,NClust)
pi,mu,cov = fit(X,mu,pi,cov,NClust,100)


# In[31]:


labels = predict(X,mu,pi,cov,NClust)
plt.scatter(X[:, 0], X[:, 1], c=labels, label=labels, s=40, cmap='viridis');
plt.axis('square');
plt.colorbar()

