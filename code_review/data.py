
import sys

import numpy as np
import laspy
import os
from math import pi
# visualizer
import vispy.scene
from vispy.scene import visuals
# xml parsing
from xml.etree.ElementTree import parse

from scipy.misc import imsave
import cv2
# Some hyper-parameters
# These should be take apart from this code[Black, Red, Green, Blue, Yellow, Magenta, Cyan, Brown, Orange, Violet ]
COLOR_CODE = [np.array([0,0,0,1]), np.array([1, 0, 0, 1]),np.array([0, 1,0, 1]), np.array(
    [0, 0, 1, 1]), np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]),
     np.array([0, 1, 1, 1]), np.array([0.6, 0.25, 0.1, 1]),np.array([1,0.4,0,1]),np.array([0.4,0,0.8,1])]
MAX_Z_VALUE = 7480.0
FEATURE_SIZE = np.array([512, 512])
PCLOUD_SIZE = np.array([25600,25600])
#PCLOUD_SIZE = np.array([12800,12800])

N_POINTS_MIN = 200000
N_LABEL_MIN = 70
VOID_LABEL_THR = 10

class Las:
    lasSubDir = 'Las'
    labelSubDir = 'Label'

    def __init__(self, dataDir, seqName):
        self.lasDir = os.path.join(dataDir, self.lasSubDir, seqName + '.las')
        self.labelDir = os.path.join(
            dataDir, self.labelSubDir, seqName + '_LAS_차선.xml') #.xml
        self.las, self.offset = self.load_las()
        self.label, self.labelForSample = self.load_label()

    def load_las(self):
        lasData = laspy.file.File(self.lasDir)
        # XYZ : original, xyz : scaled (<- some quantization problem)
        X = lasData.X
        Y = lasData.Y
        Z = lasData.Z
        I = lasData.Intensity  # uint16
        # Normalizing
        # X_norm = (X * X_scale) + X_offset
        
        X_offset, Y_offset, Z_offset = lasData._header.offset
        X_scale, Y_scale, Z_scale = lasData._header.scale
        
        xyzOffset = {'X_offset' : X_offset, 'Y_offset' : Y_offset,'Z_offset' : Z_offset,
            'X_scale' : X_scale, 'Y_scale' : Y_scale, 'Z_scale' : Z_scale}

        I = np.float32(I / 65535.0) #
        # XYZ.shape : #points x 4(XYZI)
        xyzi = np.vstack((X, Y, Z, I)).transpose()

        return xyzi, xyzOffset
    # def load_label():

    # At first, only for lane label
    def load_label(self):
        # label : lane class, type : single/double lane
        # value : lane color
        offset = self.offset
        # xml parse
        labelRaw = parse(self.labelDir)
        labelRoot = labelRaw.getroot()

        objectsStr = []  # Objects with structure
        objectsAll = []  # Objects untangled with class ( Only position )
        # loop for all index of lanes
        for obj in labelRoot.iter("Object"):
            objects = type('', (), {})()

            # Parsing for information
            info = obj.find('Information').attrib
            objects.id = int(info['id'])
            objects.label = int(info['class']) # 0 for background
            objects.type = int(info['type'])
            objects.value = int(info['value'])
            objects.nPoints = int(info['point_number'])

            if objects.label == 9: #crosswalk
                objects.label =0

            objXYZ = []
            xList = obj.find('x_location').attrib
            yList = obj.find('y_location').attrib
            zList = obj.find('z_location').attrib
            nLabelPoints = 0
            for iter_n in range(objects.nPoints):
                # since 1-based index of xml
                xPos = np.float32(xList['x{}'.format(iter_n + 1)])
                yPos = np.float32(yList['y{}'.format(iter_n + 1)])
                zPos = np.float32(zList['z{}'.format(iter_n + 1)])
                
                # coordinate transform to original pointcloud
                xPos = (xPos - offset['X_offset']) / offset['X_scale']
                yPos = (yPos - offset['Y_offset']) / offset['Y_scale']
                zPos = (zPos - offset['Z_offset']) / offset['Z_scale']

                xyziTmp = np.array([xPos, yPos, zPos, objects.label, objects.id])
                if len(objXYZ) == 0 or np.any(xyziTmp[0:3] != objXYZ[-1]):
                    objXYZ.append(xyziTmp[0:3]) 
                    objectsAll.append(xyziTmp)
                    nLabelPoints = nLabelPoints + 1

            objects.position = objXYZ
            objectsStr.append(objects)
        return objectsStr, objectsAll

    def visualize_las(self, flag_label=False):
        # create a rendering window and renderer
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()

        # data (Point cloud)
        position = self.las[:, 0:3]
        intensity = self.las[:, 3]
        nPoints = position.shape[0]
        colors = np.ones((nPoints, 4))
        # Only show intensity
        colors[:, 3] = colors[:, 3] * intensity
        
        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(position, edge_color=None, face_color=colors, size=1)
        view.add(scatter)
        
        # data (Label)
        if flag_label:
            label = self.label

            for iterId in range(len(label)):
                scatter = visuals.Markers()
                labelOne = label[iterId]
                position = np.vstack(labelOne.position)
                if position.shape[0] > 1 :
                    scatter.set_data(position, edge_color=None,
                                    face_color=COLOR_CODE[labelOne.label], size=3)
                    view.add(scatter)
        
        # setting visualizer
        view.camera = 'turntable'
        axis = visuals.XYZAxis(parent=view.scene)

        import sys
        if sys.flags.interactive != 1:
            vispy.app.run()
        return 1

    def preprocess_lane(self,maxZValue=MAX_Z_VALUE):
        las = self.las
        lab = self.labelForSample
        maxZValue = np.amax(lab,axis=0)[2] + 1000
        minZValue = np.amin(lab,axis=0)[2] - 1000
        mask = np.where((las[:, 2] < maxZValue)& (las[:,2]>minZValue))
        las = las[mask]

        self.las = las

        return True

class DataSampling:
    def __init__(self, Las, pcloudSize=PCLOUD_SIZE ,featureSize=FEATURE_SIZE):
        self.xyzi = Las.las
        self.label = np.vstack(Las.labelForSample)
        self.numLabel = self.label.shape[0]
 
        self.pcloudSize = pcloudSize
        self.featureSize = featureSize
        self.halfPc = pcloudSize // 2
        self.halfFeat = featureSize // 2

        while(1):
            self.featureMap, self.labelMap, self.labelVec, bad_flag = self.pcloud_to_feature()
            if bad_flag == False:
                break

    def select_center_point(self):
        # Rules to select center point of feature map
        halfPc = self.halfPc
        halfFeat = self.halfFeat

        max_px, max_py, _, _ = np.amax(self.xyzi, axis=0)
        min_px, min_py, _, _ = np.amin(self.xyzi, axis=0)
    
        max_lx, max_ly, _, _,_ = np.amax(self.label, axis=0)
        min_lx, min_ly, _, _,_ = np.amin(self.label, axis=0)

        while(1):
            idx = np.random.randint(self.numLabel)
            labelTmp = self.label[idx,:]
            
            # Rules,
            # (x, y, z) => (h, w, c) xy-coordinate order must be checked
            # assumes no infinite loop
            if (labelTmp[0] - halfPc[0] > min_px and labelTmp[0] + halfPc[0] < max_px
                and labelTmp[1] - halfPc[1] > min_py and labelTmp[1] + halfPc[1] < max_py):        
            
                if (labelTmp[0] - halfPc[0] > min_lx and labelTmp[0] + halfPc[0] < max_lx
                    and labelTmp[1] - halfPc[1] > min_ly and labelTmp[1] + halfPc[1] < max_ly):        
            
                    center = labelTmp[0:2]        
                    break
        return center
    def random_rotate_pcloud(self, center):
        xyzi = self.xyzi
        label = self.label

        theta = np.random.uniform(0, 2*pi)
        transform = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        
        xy_pc = xyzi[:,0:2]
        xy_lab = label[:,0:2]

        xy0 = np.tile(center,(xy_pc.shape[0],1))
        xyNew_pc = (np.dot(transform,(xy_pc-xy0).transpose()) + xy0.transpose()).transpose()
        xy0 = np.tile(center,(xy_lab.shape[0],1))
        xyNew_lab = (np.dot(transform,(xy_lab-xy0).transpose()) + xy0.transpose()).transpose()
        
        xyzi[:,0:2] = xyNew_pc
        label[:,0:2] = xyNew_lab

        self.xyzi = xyzi
        self.label = label

        return 1

    def crop_pcloud(self):
        # crop the point cloud by pre-defined offset
        # unit in meter
        
        while(1):
            center = self.select_center_point()
            self.random_rotate_pcloud(center)

            xyzi = self.xyzi
            label = self.label
            halfPc = self.halfPc
            halfFeat = self.halfFeat       

            left = center[0] - halfPc[0]
            right = center[0] + halfPc[0]
            top = center[1] - halfPc[1]
            bot = center[1] + halfPc[1]
            
            mapBoundary = {'left' : left, 'right' : right, 'top' : top, 'bot' : bot}

            mask = np.where((xyzi[:,0] >= left) & (xyzi[:,0] < right) & (xyzi[:,1] >= top) & (xyzi[:,1] < bot))
            pcloud = xyzi[mask]

            mask = np.where((label[:,0] >= left) & (label[:,0] < right) & (label[:,1] >= top) & (label[:,1] < bot))
            label_crop = label[mask]
            print('done', label_crop.shape[0])
            if pcloud.shape[0] > N_POINTS_MIN and label_crop.shape[0] > N_LABEL_MIN:
                break

        return pcloud, label_crop, mapBoundary
    
    def pcloud_to_feature(self):
        # discretize the 3d point cloud to 2d feature map
        # unit in pixel
        # feature map : (H x W x Ch), label Map : (H x W x 1)
        # ch : height(z) / Intensity
        nChannel = 2
        pcloud, label, mapBound = self.crop_pcloud()
        print('#Points in cropped region: {}'.format(pcloud.shape[0]))
        print('#Labels in cropped region: {}'.format(label.shape[0]))

        # Initialize zeros to feature map & label map
        featureMap = np.zeros(shape=(np.append(self.featureSize,nChannel))) 
        labelMap = np.zeros(shape=(np.append(self.featureSize,1)))

        # Quantize point cloud to feature map grid (x,y)
        px = pcloud[:,0]
        py = pcloud[:,1]    
        pz = pcloud[:,2]

        qx = ((px - mapBound['left']) * self.featureSize[0] // self.pcloudSize[0]).astype(np.int32)
        qy = ((py - mapBound['top']) * self.featureSize[1] // self.pcloudSize[1]).astype(np.int32)        
        

        pcloud[:,0] = qx
        pcloud[:,1] = qy
        # Same quantization for lable map
        lx = label[:,0]
        ly = label[:,1]

        qlx = ((lx - mapBound['left']) * self.featureSize[0] // self.pcloudSize[0]).astype(np.int32)
        qly = ((ly - mapBound['top']) * self.featureSize[1] // self.pcloudSize[1]).astype(np.int32)        
                
        # sort increasing order from z->y->x, use stable(lex) sort
        indices = np.lexsort((-pcloud[:,2],pcloud[:,1],pcloud[:,0]))
        pcloud = pcloud[indices, :]
        
        # Remove the point with same (x,y,z) value, remaining the maximum value
        _, indices = np.unique(pcloud[:,0:2], axis=0, return_index=True) 
        pcloud = pcloud[indices]

        # map point cloud to feature map
        featureMap[np.int_(pcloud[:,0]), np.int_(pcloud[:,1]), 0] = (pcloud[:,2] - np.amin(pcloud[:,2])) / (np.amax(pcloud[:,2]) - np.amin(pcloud[:,2]))
        featureMap[np.int_(pcloud[:,0]), np.int_(pcloud[:,1]), 1] = pcloud[:,3] 
        #print(np.amin(pcloud[:,2]))
        labelMap[qlx,qly,0] = label[:,3]
        
        labelVec = label.copy()
        labelVec[:,0] = qlx
        labelVec[:,1] = qly # What about z
        
        labelMap, bad_flag = self.remove_incorrect_label(featureMap, labelMap)

        return featureMap, labelMap, labelVec, bad_flag
    
    def remove_incorrect_label(self, featureMap, labelMap):
        # remove the end-point label if no pixel value on it
        mask = np.where((labelMap[:,:,0] != 0) & (featureMap[:,:,1] == 0))
        labelMap[mask] = 0
        if mask[0].shape[0] < VOID_LABEL_THR:
            bad_flag = False
        else:
            bad_flag = True
        return labelMap, bad_flag                                                                                                                                                                  
    def compute_dense_label(self):
        featureMap = self.featureMap
        labelMap = self.labelMap
        labelVec = self.labelVec

class FeatureWriter():
    def __init__(self, feature, outDir):
        self.featureMap = feature.featureMap
        self.labelMap = feature.labelMap
        self.labelVec = feature.labelVec
        self.pcloudSize = feature.pcloudSize
        self.featureSize = feature.featureSize
        boxWidthBias = 400
        self.pointDrawBias = 3
        self.box_width = boxWidthBias * self.featureSize[0] // self.pcloudSize[0]
        self.outDir = outDir
        self.intensityThr= [0, 0.04, 0.04]
        self.flag_refine = False # move this to
        self.flag_line = False
    # grey image to 3-channel grey image
    def grey_to_rgb(self, img_grey):
        img_rgb = np.zeros((img_grey.shape[0], img_grey.shape[1],3))
        img_rgb[:, :, 0] = img_grey
        img_rgb[:, :, 1] = img_grey
        img_rgb[:, :, 2] = img_grey

        return img_rgb
    def image_init(self):
        # Initialize feature map to image-style output for imsave to visualize
        img_tmp = self.featureMap[:,:,1]
        # gray image to 3-channel gray image
        img = self.grey_to_rgb(img_tmp)

        img_label = self.labelMap
        pos_y, pos_x,_ = np.where(img_label != 0)
        pointDrawBias = self.pointDrawBias
        # Should optimize this to no-loop version
        for iter in range(pos_x.shape[0]):
            img[pos_y[iter]:pos_y[iter]+pointDrawBias,pos_x[iter]:pos_x[iter]+pointDrawBias,:] = COLOR_CODE[np.int_(img_label[pos_y[iter],pos_x[iter]])[0]][0:3]

        return img
    # function for outputs imsave
    def write(self,seqName,flag_label):
        if flag_label == True:
            # save label map visualization & true label map
            # augmented label map with color coded label
            img = self.image_init()
            imsave(self.outDir+seqName+'_label_vis.png', img)
            #true label map
            img = np.uint8(self.labelMap[:,:,0])
            np.save(self.outDir+seqName+'_label.npy',img)
            
        else: 
            # save height map & intensity map
            # intensity map
            img = self.featureMap[:,:,1]
            imsave(self.outDir+seqName+'_intensity.png', img)
            np.save(self.outDir+seqName+'_intensity.npy',img)
            # label map
            img = self.featureMap[:,:,0]
            imsave(self.outDir+seqName+'_height.png', img)
            np.save(self.outDir+seqName+'_height.npy',img)
        return 1 
    # draw correspoinding class line between two label points which included in same id
    def draw_line(self):
        featureMap = self.featureMap
        labelMap = self.labelMap
        labelVec = np.int_(self.labelVec[:,[0,1,3,4]]) # No z-value

        print('### DO NOT COUNT IN LABEL 9 (CROSSWALK) ###')

        
        for iter in range(labelVec.shape[0] - 1): 
            # compute intensities of each line end-point
            intensity1 = featureMap[labelVec[iter,0],labelVec[iter,1],1]
            intensity2 = featureMap[labelVec[iter+1,0],labelVec[iter+1,1],1]
            
            if labelVec[iter, 2] == labelVec[iter+1, 2] and labelVec[iter,3]==labelVec[iter+1,3]: #  (intensity1>0 and intensity2>0)
                classLabel = int(labelVec[iter,2])
                labelMapBef = labelMap.copy()
                labelMap = cv2.line(labelMap,tuple(labelVec[iter,[1,0]]), tuple(labelVec[iter + 1,[1,0]]), (classLabel,classLabel,classLabel),1)
                
                thickLine = np.zeros(labelMap.shape,dtype=labelMap.dtype)
                thickLine = cv2.line(thickLine,tuple(labelVec[iter,[1,0]]), tuple(labelVec[iter + 1,[1,0]]), (classLabel,classLabel,classLabel),5)
                # apply grabcut aglorithm to each line segment
                dist = np.linalg.norm(labelVec[iter+1,:2]-labelVec[iter,:2])
                #print('pixel dist:',dist)
        
                if self.flag_refine == True and dist > 1:
                    # record only the updated line segment label for grabcut refinement
                    labelMapDelta = thickLine - labelMapBef
                    #print('num Lane pixels:',labelMapDelta[labelMapDelta>0].shape[0])
                    mask = np.where(thickLine>0)
                    intensityMap = featureMap[:,:,1].reshape(np.append(self.featureSize,1))
                    labelMapDelta[intensityMap< 3.5*self.intensityThr[classLabel]] = 0
                    intensityMap = intensityMap[mask]
                    avgIntensity = np.average(intensityMap)
                    #print('label & average intensity:', classLabel, avgIntensity)
                    if labelMapDelta[labelMapDelta>0].shape[0] >4 and avgIntensity>self.intensityThr[classLabel]:
                        labelRefined = self.refine_label(featureMap, labelMapDelta, labelVec[iter], labelVec[iter+1],classLabel)
                        labelMap = np.maximum(labelRefined, labelMapBef)
                    else:
                        labelMap = labelMapBef
                    if np.sum(labelMap[labelVec[iter,0]-1:labelVec[iter,0]+2 ,labelVec[iter,1]-1:labelVec[iter,1]+2]) ==classLabel:
                         labelMap[labelVec[iter,0],labelVec[iter,1]] = 0
                elif self.flag_line==True:
                   labelMap=labelMap
                else:
                    labelMap = labelMapBef
       # self.image
        self.labelMap = labelMap        
        return 1
    
    # refine the line label to actual road lane by grabcut algorithm
    def refine_label(self, featureMap, labelMapLine, vec1, vec2, label):
        iter_grabcut = 5
        y1 = vec1[0]
        x1 = vec1[1]
        y2 = vec2[0]
        x2 = vec2[1]
        pos12 = np.array([[y1,x1],[y2,x2]])
        box_width = self.box_width
        
        # Crop corresponding rectangular region (need to width of l)
        # m : perpendicular slope of (x1, y1) to (x2, y2)
        if x1 == x2:
            m = 0
        elif y1 == y2:
            m = None
        else:
            m = -1 / ((y2 - y1) / (x1- x2))
        # posTight : tight position of lane box
        posTight = np.zeros((4,2),dtype=int)
        # loop for (y1,x1) and (y2,x2)
        for it_pos in range(2):
            # loop for direction
            for direction in [-1, 1]:
                if m ==0:
                    deltaY = 0
                    deltaX = direction * box_width / 2
                elif m == None:
                    deltaY = direction * box_width / 2
                    deltaX = 0
                else: 
                    deltaY = direction * (box_width / 2) * m / np.sqrt(m**2 + 1) 
                    deltaX = direction * (box_width / 2) * 1 / np.sqrt(m**2 + 1) 
                bias = it_pos * 2 + int(0.5 * direction + 0.5)
                posTight[bias,:] = pos12[it_pos,:] + np.array([deltaY, deltaX])
        # compute stand straight box
        
        max_by, max_bx = np.amax(posTight, axis=0)
        min_by, min_bx = np.amin(posTight, axis=0)
        rect = (min_bx, min_by, max_bx - min_bx, max_by - min_by) # (x,y,w,h)
        
        img = self.grey_to_rgb(featureMap[:,:,1])
        #img[:,:,0] = featureMap[:,:,0]
        img = (255 * img).astype(np.uint8)
        #imsave('tmp_0.png',img)
        # only rect
        labelMapLine = labelMapLine.reshape(labelMapLine.shape[0:2])
        #img[labelMapLine>0] = 0.2
        mask = np.zeros(labelMapLine.shape, np.uint8)
        
    
        """
        print(posTight)
        print('pixel position:',vec1[:2], vec2[:2])
        print ("num pixel line:",labelMapLine[labelMapLine>0].shape)        
        print('m, rect:', m, rect)
        """
        
    
        bgdModel = np.zeros((1,65),np.float64) # just default model
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(img, mask, rect,bgdModel,fgdModel,
            1,cv2.GC_INIT_WITH_RECT) #cv2.,GC_INIT_WITH_MASK GC_INIT_WITH_RECT

        # update with mask

        mask[labelMapLine > 0] = 1      
        #print ("num pixel line:",labelMapLine[labelMapLine>0].shape)        
    
        bgdModel = np.zeros((1,65),np.float64) # just default model
        fgdModel = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel= cv2.grabCut(img, mask, rect,bgdModel,fgdModel,
            1,cv2.GC_INIT_WITH_MASK) #cv2.,GC_INIT_WITH_MASK GC_INIT_WITH_RECT
        mask2 = np.where((mask==1)|(mask==3) ,label,0).reshape(np.append(mask.shape,1))   
    
        # mark background with intensity is zero
     
        #print('done')

        return mask2

        # give background mask for outter region for skew rectangle region
# Just test the code
if __name__ == '__main__':
    # directory define
    dataDir = '/home/dongwoo/Project/dataset/SKT/Lane/'
    #featureDir = './feature/refined_{}_{}_{}/'.format(PCLOUD_SIZE[0],FEATURE_SIZE[0],N_POINTS_MIN)
    featureDir = './data/'

    counter = 0#615#417
    if not os.path.isdir(featureDir):
        os.mkdir(featureDir)
        os.mkdir(featureDir+'feature/')
        os.mkdir(featureDir+'label/')
    while(1):
        for seqIter in os.listdir(dataDir+'Las'):
            seqName = os.path.splitext(seqIter)[0]
            print(seqName)
            # input
            seq = Las(dataDir,seqName)
            seq.preprocess_lane()    
            # point cloud visualization
            #if seqName == 'Track_A_Track_A_20161205_032536 Profiler.zfs_14':
                #seq.visualize_las(flag_label=True)
            feature = DataSampling(seq, PCLOUD_SIZE, FEATURE_SIZE)
            
            # save input bird-eye view map
            writer = FeatureWriter(feature, featureDir)
            writer.write('feature/{}'.format(counter),flag_label=False)
            writer.draw_line()
            writer.write('label/{}'.format(counter),flag_label=True)
            counter = counter + 1

            writer = FeatureWriter(feature, featureDir)
            writer.flag_line=True
            writer.draw_line()
            writer.write('label/{}'.format(counter),flag_label=True)
            counter = counter + 1
            
            writer = FeatureWriter(feature, featureDir)
            writer.flag_refine=True
            writer.pointDrawBias = 1
            writer.draw_line()
            writer.write('label/{}'.format(counter),flag_label=True)
            counter = counter + 1
        if counter>1000:
            break
        """qwe
        writer = FeatureWriter(feature, featureDir)
        writer.write(seqName+'_point')
        writer.draw_line()
        writer.write(seqName+'_line')
        """

