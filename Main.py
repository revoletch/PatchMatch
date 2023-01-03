import random
import timeit
import scipy.io as sci
import numpy as num
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from math import log10,sqrt

def PSNR(ori, comp) :
    mse = num.mean((ori-comp) ** 2)
    if(mse == 0):
        return 100
    maxPixel = 512
    psnr = 20 * log10(maxPixel / sqrt(mse))
    return psnr

t_start = timeit.default_timer()

# define the patch size
halfPatchSize = 1
# numIte = int(input('set the iteration: ').strip())  # can be improved into asking input
numIte= 8

patchSize = [2*halfPatchSize+1, 2*halfPatchSize+1]  # [x,y]

# %% Data Import and Editing

# import data
atlas = sci.loadmat('UKMainzTestImages.mat')
# extract atlas for CT and CBCT from the Atlas + random sliceNum
#sliceNum = random.randint(70, 168)
sliceNum = int(156)
atlasCT = atlas['CTvol'][:, :, sliceNum]
atlasCT = atlasCT / num.amax(atlasCT)
atlasCBCT = atlas['warpedCBCT'][:, :, sliceNum]
atlasCBCT = atlasCBCT / num.amax(atlasCBCT)

# creating slice of-the-day
targetsliceNum = sliceNum + random.choice([3, -3])

currCT = atlas['CTvol'][:, :, targetsliceNum]
currCBCT = atlas['warpedCBCT'][:, :, targetsliceNum]


# rotAngle = random.choice([-5, 5])
rotAngle = int(5)
currCT = rotate(currCT, angle=rotAngle)
currCT = currCT / num.amax(currCT)
currCBCT = rotate(currCBCT, angle=rotAngle)
currCBCT = currCBCT / num.amax(currCBCT)
# resize slice OTD
resizeFac = len(atlasCT[:, 0]) / len(currCT[:, 0])
currCT = zoom(currCT, resizeFac)
currCBCT = zoom(currCBCT, resizeFac)

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
# plt.show()
# plt.figure()
# plt.imshow(currCT)
# plt.gray()
# plt.title('currCT')
# plt.show()
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
tot = 0
# loop Iteration
for it in range(numIte):
    # debug
    tot+=1
    ##
    PSNRiter = it
    shuffledXX = num.reshape(XXvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))
    shuffledYY = num.reshape(YYvec[num.random.permutation(imsizex*imsizey)], (imsizey, imsizex))
    RSF[:, :, 0] = shuffledXX - XX
    RSF[:, :, 1] = shuffledYY - YY
    propamask = num.zeros(num.shape(currCBCT))

    # Phase II: Propagation of the good matches to the right and bottom
    startx = halfPatchSize + 2
    endx = imsizex - halfPatchSize
    starty = halfPatchSize + 2
    endy = imsizey - halfPatchSize

    for j in range(starty, endy):
        for i in range(startx, endx):
            uA = int(min(max(i+NNF[j, i, 0], halfPatchSize+1), imsizex-halfPatchSize))
            uB = int(min(max(i+NNF[j, i-1, 0], halfPatchSize+1), imsizex-halfPatchSize))
            uC = int(min(max(i+NNF[j-1, i, 0], halfPatchSize+1), imsizex-halfPatchSize))
            vA = int(min(max(i+NNF[j, i, 1], halfPatchSize+1), imsizey-halfPatchSize))
            vB = int(min(max(i+NNF[j, i-1, 1], halfPatchSize+1), imsizey-halfPatchSize))
            vC = int(min(max(i+NNF[j-1, i, 1], halfPatchSize+1), imsizey-halfPatchSize))

            if halfPatchSize == 1:
                if uA == 511:
                    uA = 510
                if vA == 511:
                    vA = 510
                if uB == 511:
                    uB = 510
                if vB == 511:
                    vB = 510
                if uC == 511:
                    uC = 510
                if vC == 511:
                    vC = 510
            elif halfPatchSize == 2:
                if uA == 510:
                    uA = 509
                if vA == 510:
                    vA = 509
                if uB == 510:
                    uB = 509
                if vB == 510:
                    vB = 509
                if uC == 510:
                    uC = 509
                if vC == 510:
                    vC = 509

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
                g = int(min(max(umin + num.round((0.5**k) *RSF[vmin, umin, 0]), halfPatchSize+1), imsizex-halfPatchSize))
                h = int(min(max(umin + num.round((0.5**k) *RSF[vmin, umin, 1]), halfPatchSize+1), imsizey-halfPatchSize))

                if halfPatchSize == 1:
                    if h == 511:
                        h = 510
                    if g == 511:
                        g = 510
                elif halfPatchSize == 2:
                    if h == 510:
                        h = 509
                    if g == 510:
                        g = 509

                testPatchD = atlasCBCT[h-halfPatchSize:h +halfPatchSize+1, g-halfPatchSize:g+halfPatchSize+1]
                distD = num.sum((currPatch-testPatchD)**2)
                if distD < mindist:
                    NNF[j, i, 0] = g-i
                    NNF[j, i, 1] = h-j

    # Phase IV: Propagation of the good matches to the left up
    for j in range(endy-1, starty-1, -1):
        for i in range(endy-1, starty-1, -1):
            uA = int(min(max(i+NNF[j, i, 0], halfPatchSize+1), imsizex-halfPatchSize))
            uB = int(min(max(i+NNF[j, i+1, 0], halfPatchSize+1), imsizex-halfPatchSize))
            uC = int(min(max(i+NNF[j+1, i, 0], halfPatchSize+1), imsizex-halfPatchSize))
            vA = int(min(max(i+NNF[j, i, 1], halfPatchSize+1), imsizey-halfPatchSize))
            vB = int(min(max(i+NNF[j, i+1, 1], halfPatchSize+1), imsizey-halfPatchSize))
            vC = int(min(max(i+NNF[j+1, i, 1], halfPatchSize+1), imsizey-halfPatchSize))

            if halfPatchSize == 1:
                if uA == 511:
                    uA = 510
                if vA == 511:
                    vA = 510
                if uB == 511:
                    uB = 510
                if vB == 511:
                    vB = 510
                if uC == 511:
                    uC = 510
                if vC == 511:
                    vC = 510
            elif halfPatchSize == 2:
                if uA == 510:
                    uA = 509
                if vA == 510:
                    vA = 509
                if uB == 510:
                    uB = 509
                if vB == 510:
                    vB = 509
                if uC == 510:
                    uC = 509
                if vC == 510:
                    vC = 509

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
                g = int(min(max(umin + num.round((0.5**k) *RSF[vmin, umin, 0]), halfPatchSize+1), imsizex-halfPatchSize))
                h = int(min(max(umin + num.round((0.5**k) *RSF[vmin, umin, 1]), halfPatchSize+1), imsizey-halfPatchSize))

                if halfPatchSize == 1:
                    if h == 511:
                        h = 510
                    if g == 511:
                        g = 510
                elif halfPatchSize == 2:
                    if h == 510:
                        h = 509
                    if g == 510:
                        g = 509

                testPatchD = atlasCBCT[h-halfPatchSize:h +halfPatchSize+1, g-halfPatchSize:g+halfPatchSize+1]
                distD = num.sum((currPatch-testPatchD)**2)
                if distD < mindist:
                    NNF[j, i, 0] = g-i
                    NNF[j, i, 1] = h-j
    # # image of Propamask
    # plt.figure()
    # plt.imshow(propamask)
    # plt.gray()
    # plt.title('Propamask, Iteration: ' + str(it))
    # # image of NNF1
    # plt.figure()
    # plt.imshow(NNF[:,:,0])
    # plt.gray()
    # plt.title('NNF1, Iteration: ' + str(it))
    # # image of NNF2
    # plt.figure()
    # plt.imshow(NNF[:,:,1])
    # plt.gray()
    # plt.title('NNF2, Iteration: ' + str(it))
    # Generation of the synthetic image
    
    synthCBCT = num.zeros(num.shape(currCBCT))
    synthCT = num.zeros(num.shape(currCBCT))

    for s in range(startx, endx):
        for t in range(starty, endy):
            pick_x = int(min(max(s+NNF[t, s, 0], 1), imsizex-1))
            pick_y = int(min(max(t+NNF[t, s, 1], 1), imsizey-1))

            synthCBCT[t, s] = atlasCBCT[pick_x, pick_y]
            synthCT[t, s] = atlasCT[pick_x, pick_y]
    # # PSNR calculation
    if PSNRiter != numIte:
        CBCTpsnr[PSNRiter] = PSNR(currCBCT, synthCBCT)
        # CTpsnr = PSNR(currCT, synthCT)
    
    # image synth CBCT
    plt.figure()
    plt.imshow(synthCBCT)
    plt.gray()
    plt.title('synth CBCT, Iteration: ' + str(it))
    # # image synth CT
    # plt.figure()
    # plt.imshow(synthCT)
    # plt.gray()
    # plt.title('synth CT, Iteration: ' + str(it))

stop = timeit.default_timer()
print(f'Time: {stop-t_start}')
