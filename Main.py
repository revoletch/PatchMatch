import random
import timeit
import scipy.io as sci
import numpy as num
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from skimage import metrics as mcs

def maxIndex(a,b):
    a = int(a)
    test = a
    if test<b:
        test = b
    return test

def minIndex(a,b):
    a = int(a)
    test = a
    if test>b:
        test = b
    return test

t_start = timeit.default_timer()

#define resize factor
resizeFac = 0.25
# define the patch size
halfPatchSize = 2
# iteration number
numIte= 8

patchSize = [2*halfPatchSize+1, 2*halfPatchSize+1]  # [x,y]

# %% Data Import and Editing

# import data
atlas = sci.loadmat('UKMainzTestImages.mat')
# extract atlas for CT and CBCT from the Atlas + random sliceNum
# sliceNum = random.randint(70, 168)
sliceNum = 139
atlasCT = atlas['CTvol'][:, :, sliceNum]
atlasCT = atlasCT / num.amax(atlasCT)
atlasCBCT = atlas['warpedCBCT'][:, :, sliceNum]
atlasCBCT = atlasCBCT / num.amax(atlasCBCT)

# creating slice of-the-day
targetsliceNum = sliceNum + random.choice([5, -5])

currCT = atlas['CTvol'][:, :, targetsliceNum]
currCBCT = atlas['warpedCBCT'][:, :, targetsliceNum]



# rotAngle = random.choice([-5, 5])
rotAngle = 5
currCT = rotate(currCT, angle=rotAngle, reshape = False)
currCT = currCT / num.amax(currCT)
currCBCT = rotate(currCBCT, angle=rotAngle, reshape = False)
currCBCT = currCBCT / num.amax(currCBCT)
# resize the slices
currCT = zoom(currCT, resizeFac)
currCBCT = zoom(currCBCT, resizeFac)
atlasCT = zoom(atlasCT, resizeFac)
atlasCBCT = zoom(atlasCBCT, resizeFac)

# generate the images
# https://docs.scipy.org/doc/scipy/tutorial/signal.html)
# plt.figure()
# plt.imshow(atlasCT)
# plt.gray()
# plt.title('Atlas CT')
# plt.show()
# plt.figure()
# plt.imshow(atlasCBCT)
# plt.gray()
# plt.title('Atlas CBCT')
# #CBCT showing HU deviation which could cause unnecessary radiation on the patient
plt.show()
plt.figure()
plt.imshow(currCT)
plt.gray()
plt.title('currCT')
plt.show()
plt.figure()
plt.imshow(currCBCT)
plt.gray()
plt.title('currCBCT')
plt.show()

# %% Patch Match
imsizex = len(atlasCT[:, 0])
imsizey = len(atlasCT[0, :])
# PSNR Vectors
CTpsnr = num.zeros(numIte)
CBCTpsnr = num.zeros(numIte)
# Phase I: Initialization of  NNF w/ random offsets
NNF = num.zeros([imsizex, imsizey, 2])
RSF = num.zeros([imsizex, imsizey, 2])
# random offset
XX, YY = num.meshgrid(num.arange(0, imsizex), num.arange(0, imsizey))
XXvec = XX.flatten()
YYvec = YY.flatten()
shuffledXX = num.reshape(XXvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))
shuffledYY = num.reshape(YYvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))

NNF[:, :, 0] = shuffledXX - XX
NNF[:, :, 1] = shuffledYY - YY

# preinitialize the patches
currPatch = num.zeros(patchSize)
testPatchA = num.zeros(patchSize)
testPatchB = num.zeros(patchSize)
testPatchC = num.zeros(patchSize)
testPatchD = num.zeros(patchSize)
PSNRiter = int('0')


#%% loopy loop

for it in range(numIte):
    print(f'going through {it+1}. Iteration')
    PSNRiter = it
    
    shuffledXX = num.reshape(XXvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))
    shuffledYY = num.reshape(YYvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))
    RSF[:, :, 0] = shuffledXX - XX
    RSF[:, :, 1] = shuffledYY - YY
    propamask = num.zeros(num.shape(currCBCT))

    # Phase II: Propagation of the good matches to the right and bottom
    startx = halfPatchSize
    endx = imsizex - halfPatchSize - 1
    starty = halfPatchSize
    endy = imsizey - halfPatchSize - 1 

    for j in range(starty, endy):
        for i in range(startx, endx):
            uA = int(minIndex(maxIndex(i+NNF[j, i, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            uB = int(minIndex(maxIndex(i+NNF[j, i-1, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            uC = int(minIndex(maxIndex(i+NNF[j-1, i, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            vA = int(minIndex(maxIndex(j+NNF[j, i, 1], halfPatchSize), imsizey-halfPatchSize - 1))
            vB = int(minIndex(maxIndex(j+NNF[j, i-1, 1], halfPatchSize), imsizey-halfPatchSize - 1))
            vC = int(minIndex(maxIndex(j+NNF[j-1, i, 1], halfPatchSize), imsizey-halfPatchSize - 1))
            
            currPatch = currCBCT[j-halfPatchSize: j +halfPatchSize+1, i-halfPatchSize: i+halfPatchSize+1]
            testPatchA = atlasCBCT[vA-halfPatchSize:vA +halfPatchSize+1, uA-halfPatchSize:uA+halfPatchSize+1]
            testPatchB = atlasCBCT[vB-halfPatchSize:vB +halfPatchSize+1, uB-halfPatchSize:uB+halfPatchSize+1]
            testPatchC = atlasCBCT[vC-halfPatchSize:vC +halfPatchSize+1, uC-halfPatchSize:uC+halfPatchSize+1]

            distA = num.sum((currPatch-testPatchA)**2)
            distB = num.sum((currPatch-testPatchB)**2)
            distC = num.sum((currPatch-testPatchC)**2)
            mindist = distA
            umin = uA
            vmin = vA

            if (distC < distA) and (distC < distB):
                NNF[j, i, :] = NNF[j-1, i, :]
                mindist = distC
                umin = uC
                vmin = vC
            elif distB < distA:
                NNF[j, i, :] = NNF[j, i-1, :]
                mindist = distB
                umin = uB
                vmin = vB

            # debug info
            if mindist < distA:
                propamask[j, i] = 1

            # Phase III: Random search phase
            for k in range(0, 8):
                g = int(minIndex(maxIndex(umin + num.round((0.5**k) *RSF[vmin, umin, 0]), halfPatchSize), imsizex-halfPatchSize - 1))
                h = int(minIndex(maxIndex(vmin + num.round((0.5**k) *RSF[vmin, umin, 1]), halfPatchSize), imsizey-halfPatchSize - 1))

                testPatchD = atlasCBCT[h-halfPatchSize:h +halfPatchSize+1, g-halfPatchSize:g+halfPatchSize+1]
                distD = num.sum((currPatch-testPatchD)**2)
                if distD < mindist:
                    NNF[j, i, 0] = g-i
                    NNF[j, i, 1] = h-j

    # Phase IV: Propagation of the good matches to the left up
    for j in range(endy, starty-1, -1):
        for i in range(endx, startx-1, -1):
            uA = int(minIndex(maxIndex(i+NNF[j, i, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            uB = int(minIndex(maxIndex(i+NNF[j, i+1, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            uC = int(minIndex(maxIndex(i+NNF[j+1, i, 0], halfPatchSize), imsizex-halfPatchSize - 1))
            vA = int(minIndex(maxIndex(j+NNF[j, i, 1], halfPatchSize), imsizey-halfPatchSize - 1))
            vB = int(minIndex(maxIndex(j+NNF[j, i+1, 1], halfPatchSize), imsizey-halfPatchSize - 1))
            vC = int(minIndex(maxIndex(j+NNF[j+1, i, 1], halfPatchSize), imsizey-halfPatchSize - 1))

            currPatch = currCBCT[j-halfPatchSize: j +halfPatchSize+1, i-halfPatchSize: i+halfPatchSize+1]
            testPatchA = atlasCBCT[vA-halfPatchSize:vA +halfPatchSize+1, uA-halfPatchSize:uA+halfPatchSize+1]
            testPatchB = atlasCBCT[vB-halfPatchSize:vB +halfPatchSize+1, uB-halfPatchSize:uB+halfPatchSize+1]
            testPatchC = atlasCBCT[vC-halfPatchSize:vC +halfPatchSize+1, uC-halfPatchSize:uC+halfPatchSize+1]   

            distA = num.sum((currPatch-testPatchA)**2)
            distB = num.sum((currPatch-testPatchB)**2)
            distC = num.sum((currPatch-testPatchC)**2)
            mindist = distA
            umin = uA
            vmin = vA

            if (distC < distA) and (distC < distB):
                NNF[j, i, :] = NNF[j-1, i, :]
                mindist = distC
                umin = uC
                vmin = vC
            elif distB < distA:
                NNF[j, i, :] = NNF[j, i-1, :]
                mindist = distB
                umin = uB
                vmin = vB

            # debug info
            if mindist < distA:
                propamask[j, i] = 1

            # Phase III: Random search phase
            for k in range(0, 8):
                g = int(minIndex(maxIndex(umin + num.round((0.5**k) *RSF[vmin, umin, 0]), halfPatchSize+1), imsizex-halfPatchSize-1))
                h = int(minIndex(maxIndex(vmin + num.round((0.5**k) *RSF[vmin, umin, 1]), halfPatchSize+1), imsizey-halfPatchSize-1))

                testPatchD = atlasCBCT[h-halfPatchSize:h +halfPatchSize+1, g-halfPatchSize:g+halfPatchSize+1]
                distD = num.sum((currPatch-testPatchD)**2)
                if distD < mindist:
                    NNF[j, i, 0] = g-i
                    NNF[j, i, 1] = h-j
    # # image of Propamask
    # plt.figure()
    # plt.imshow(propamask)
    # plt.gray()
    # plt.colorbar()
    # plt.title('Propamask, Iteration: ' + str(it+1))
    # # image of NNF1
    # # plt.figure()
    # plt.imshow(NNF[:,:,0])
    # plt.gray()
    # plt.colorbar()
    # plt.title('NNF1, Iteration: ' + str(it+1))
    # # image of NNF2
    # # plt.figure()
    # plt.imshow(NNF[:,:,1])
    # plt.gray()
    # plt.colorbar()
    # plt.title('NNF2, Iteration: ' + str(it+1))
    
    # Generation of the synthetic image
    synthCBCT = num.zeros(num.shape(currCBCT))
    synthCT = num.zeros(num.shape(currCBCT))

    for s in range(startx, endx):
        for t in range(starty, endy):
            pick_x = int(minIndex(maxIndex(s+NNF[t, s, 0], 0), imsizex-1))
            pick_y = int(minIndex(maxIndex(t+NNF[t, s, 1], 0), imsizey-1))

            synthCBCT[t,s] = atlasCBCT[pick_y, pick_x]
            synthCT[t,s] = atlasCT[pick_y, pick_x]
    # PSNR calculation
    CTpsnr[PSNRiter] = mcs.peak_signal_noise_ratio(currCT, synthCT,data_range=1)
    CBCTpsnr[PSNRiter] = mcs.peak_signal_noise_ratio(currCBCT, synthCBCT,data_range=1)
    # image synth CBCT
    plt.figure()
    plt.imshow(synthCBCT)
    plt.gray()
    plt.title(f'synth CBCT, Iteration: {it+1}, PSNR: {num.round(CBCTpsnr[PSNRiter],decimals=2)}')
    # image synth CT
    plt.figure()
    plt.imshow(synthCT)
    plt.gray()
    plt.title(f'synth CT, Iteration: {it+1}, PSNR: {num.round(CTpsnr[PSNRiter],decimals=2)}')
    

#plot the PSNR
X = num.linspace(1,numIte, num=numIte)
plt.figure()
plt.plot(X,CTpsnr,color='red',marker = '.',linestyle='dashed')
plt.plot(X,CBCTpsnr,color='blue',marker = '.',linestyle='dashed')
plt.legend(['CT PSNR','CBCT PSNR'])
plt.title('PSNR')

    

stop = timeit.default_timer()
# counting the duration of the code
duration = stop-t_start
duration  = num.round(duration, decimals=2)
print(f'Time: {duration} s')
