# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ###  Implementation of gaussian mixture model (aka mixture of gaussians) 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import multivariate_normal as mvn
from seaborn import heatmap

if False:
    NClust = int(input("How many clusters do you want to fit? "));
else:
    NClust = 3

mName = 'LEW_002'
expDate = '2018-05-16'

dataFname = 'isodist_%s_%s.npy' % (mName,expDate)
X = np.load('../'+dataFname)
if np.any(np.isnan(X)):
    print('removing NaN rows...')
    print(np.shape(X))
    X = X[~np.isnan(X).any(axis=1)]
    print(np.shape(X))

plt.figure(0)
plt.scatter(X[:,0], X[:,1]) 
plt.axis('square');
plt.show()

# +
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
    gamma = respb / resb_norm #normalize to make it a real pdf (sums to 1)
    return gamma

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

def getLoss(X,pi,mu,cov,gamma,NClust):
    NData = X.shape[0]
    loss = np.zeros((NData,NClust))
    for cl in range(NClust):
        dist = mvn(mu[cl,:], cov[cl],allow_singular=True)
        currloss = gamma[:,cl] * (np.log(pi[cl]+0.00001)+
                                    dist.logpdf(X)-np.log(gamma[:,cl]+0.000001))
        loss[:,cl] = currloss
    finalLoss = np.sum(loss)
    return finalLoss

def fit(X,mu,pi,cov,NClust,NIters):
    itr = 0
    lastLoss = 0
    while True:
        itr += 1
        gamma  = eStep(X,mu,cov,pi,NClust)
        pi, mu, cov = mStep(X,gamma,NClust)
        loss = getLoss(X, pi, mu, cov,gamma,NClust)
        if itr % 10 == 0:
            print("Iteration: %d Loss: %0.6f" %(itr, loss))      
        if abs(loss-lastLoss) < 1e-6:
            break
        lastLoss = loss
    return pi,mu,cov

def predict(X,mu,pi,cov,NClust):
    labels = np.zeros((X.shape[0], NClust))

    for cl in range(NClust):
        labels [:,cl] = pi[cl] * mvn.pdf(X, mu[cl,:], cov[cl])
    labels  = labels .argmax(1)
    return labels 


# -

from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\FFmpeg\bin\ffmpeg.exe'
def animate(itr):
    if itr == 0:
        mu,cov,pi = initParams(X,NClust)
    gamma = eStep(X,mu,cov,pi,NClust)
    #plt.figure(figsize=(30,2))
    #heatmap(gamma.T)
    #plt.show()
    pi, mu, cov = mStep(X,gamma,NClust)
    
    labels = predict(X,mu,pi,cov,NClust)
    ax1.clear()
    ax1.scatter(X[:, 0], X[:, 1], c=labels, label=labels, s=40, cmap='viridis');
    ax1.set_aspect('equal')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
anim = animation.FuncAnimation(fig, animate, frames=1)

mu,cov,pi = initParams(X,NClust)
pi,mu,cov = fit(X,mu,pi,cov,NClust,100)

labels = predict(X,mu,pi,cov,NClust)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=labels, label=labels, s=40, cmap='viridis');
plt.axis('square');
plt.colorbar()
plt.show()
