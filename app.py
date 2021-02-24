from flask import Flask,render_template,request,flash
from werkzeug.utils import secure_filename
import os
from strUtil import Pic_str
import base64
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from skimage.measure import label,regionprops
from numba import jit

import skimage
from skimage import morphology
from sklearn.cluster import KMeans
import scipy.ndimage
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage as ndi
import scipy

app = Flask(__name__)

app.secret_key='aaabbb'
UPLOAD_FOLDER = 'upload'
BINARY_IMAGE_FOLDER='binaryImage'
app.config['BINARY_IMAGE_FOLDER']=BINARY_IMAGE_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['photo']
    if f and allowed_file(f.filename):
        fname = secure_filename(f.filename)
        #print(fname)
        ext = fname.rsplit('.', 1)[1]
        new_filename = Pic_str().create_uuid() + '.' + ext
        f.save(os.path.join(file_dir, new_filename))
        print("上传成功")
        return new_filename,f.filename
    else:
        print("上传失败")
        return 'fail'
def api_saveBinImage(picturePath,newFileName):
    file_dir = os.path.join(basedir, app.config['BINARY_IMAGE_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    I = mpimg.imread(picturePath)
    I = I.astype(np.int16)
    size1 = I.shape
    height = size1[0]
    width = size1[1]
    I1 = getBinaryImage(height, width, I)
    binFileName='b'+newFileName
    plt.imshow(I1)
    plt.axis('off')
    plt.savefig(os.path.join(file_dir, binFileName))
    return os.path.join(file_dir, binFileName)

# def getNumOfGrains(picturePath):
#     I = mpimg.imread(picturePath)
#     I = I.astype(np.int16)
#     size1 = I.shape
#     height = size1[0]
#     width = size1[1]
#     I1 = getBinaryImage(height, width, I)
#     # plt.imshow(I1)  # 显示图片
#     # plt.axis('off')  # 不显示坐标轴
#     # plt.show()
#     labeled_image = label(I1, connectivity=1)
#     props = regionprops(labeled_image)
#     each_grain_area = []
#     sort_each_grain_area = []
#     numObjects = np.max(labeled_image)
#
#     each_grain_area = [props[i].area for i in range(numObjects)]
#     sort_each_grain_area = sorted(each_grain_area)
#     a1 = sort_each_grain_area[:]
#     thr = 0.1
#     start0 = 0
#     # end0=0
#     for i in range(numObjects):
#         if (a1[i] < 100):
#             continue
#         if ((a1[i + 1] - a1[i]) / a1[i] < thr) and ((a1[i + 2] - a1[i + 1]) / a1[i + 1] < thr) and (
#                 (a1[i + 3] - a1[i + 2]) / a1[i + 2] < thr) and ((a1[i + 4] - a1[i + 3]) / a1[i + 3] < thr):
#             start0 = i
#             break
#
#     temp_list = a1[start0:start0 + 4]
#     one_grain_area = np.mean(temp_list)
#
#     totalNum = 0;
#     for k in range(numObjects):
#         temp1 = int((sort_each_grain_area[k] / one_grain_area))  # 取整数部分
#         temp2 = (sort_each_grain_area[k] / one_grain_area) - temp1
#         if (temp2 > 0.7 and temp1 == 0):
#             totalNum = totalNum + temp1 + 1
#         else:
#             totalNum = totalNum + temp1
#     return totalNum

def getBinaryImage(picturePath):
    I = mpimg.imread(picturePath)
    I = I.astype(np.int16)
    size1 = I.shape
    height = size1[0]
    width = size1[1]
    I_column = I.reshape(size1[0] * size1[1], 3)
    thr1 = 60;
    thr2 = 60;
    I1 = I_column[:, 0] - I_column[:, 2]
    I1[I1 >= thr1] = 30

    I2 = I_column[:, 1] - I_column[:, 2]
    I2[I2 >= thr2] = 30
    I3 = I_column[:, 0] - I_column[:, 1]
    I4 = I_column[:, 0] + I_column[:, 1] + I_column[:, 2]

    I1 = np.array([I1, I2]).T;

    matrix1 = np.array([[40, 40], [0, 0]])
    # print(matrix1)
    kms = KMeans(n_clusters=2, init=matrix1).fit(I1)
    idx = kms.labels_

    temp1 = np.mean(I4[idx == 0])
    temp2 = np.mean(I4[idx == 1])
    if (temp1 > temp2):
        idx = ~idx
    idx_reshape = idx.reshape(size1[0], size1[1])
    idx_reshape[idx_reshape == -1] = 1
    idx_reshape[idx_reshape == -2] = 0
    I1 = idx_reshape;

    I1 = I1 > 0
    I1 = morphology.remove_small_objects(I1, 80, connectivity=1)
    I1 = scipy.ndimage.binary_fill_holes(I1)
    labeled_image = label(I1, connectivity=1)
    props = regionprops(labeled_image)
    numObjects = np.max(labeled_image)

    return [I1,labeled_image,numObjects]

@jit()
def get_eachGrainAreaHistogram(height,width,I_gray,labeled_image,numObjects):
    each_grain_area_histogram = np.zeros((numObjects, 256))
    for i in range(height):
        for j in range(width):
            temp=labeled_image[i,j]
            if(temp!=0):
                temp1=I_gray[i,j]
                each_grain_area_histogram[temp-1, temp1] = each_grain_area_histogram[temp-1, temp1] + 1
    return each_grain_area_histogram

def fspecial_gauss(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

@jit
def getNewBinImageAndLabeled(height,width,labeled_image,each_grain_area_histogram_inflection_position,I_gray,I1):
    for i in range(height):
        for j in range(width):
            temp=labeled_image[i,j]
            if(temp!=0):
                thr=each_grain_area_histogram_inflection_position[temp-1,0]
                if(thr!=0 and I_gray[i,j]<thr):

                    I1[i,j]=0
                    labeled_image[i,j]=0
    return [I1,labeled_image]

def getProcessedBinaryImage(picturePath):
    I = mpimg.imread(picturePath)
    I = I.astype(np.int16)
    I_gray = I[:, :, 0]
    size1 = I.shape
    [I1,labeled_image,numObjects]=getBinaryImage(picturePath)
    each_grain_area_histogram = get_eachGrainAreaHistogram(size1[0], size1[1], I_gray, labeled_image,numObjects)
    each_grain_area_histogram_inflection_position = np.zeros((numObjects, 1))
    x = [i for i in range(1, 257)]
    d = 5
    initial1 = 150
    for i in range(numObjects):  # 对直方图进行平滑
        h = fspecial_gauss([1, 100], 7.5)
        tmp = np.array(each_grain_area_histogram[i, :]).reshape((1, 256))
        tmp = scipy.ndimage.correlate(tmp, h, mode='constant')

        # each_grain_area_histogram[i, :]=gaussian_filter1d(tmp,7.5)
        # result=scipy.ndimage.correlate(tmp, h, mode='nearest').transpose()
        # y=each_grain_area_histogram[i, :]
        y = tmp.tolist()[0]
        r = np.polyfit(x, y, d)
        yvals = np.polyval(r, range(initial1, 256))
        indx1 = np.argmax(yvals)
        indx1 = indx1 + initial1 - 2
        for k in reversed(list(range(2, indx1 + 1))):
            temp1 = r[0] * k ** 5 + r[1] * k ** 4 + r[2] * k ** 3 + r[3] * k ** 2 + r[4] * k + r[5]
            temp2 = r[0] * (k - 1) ** 5 + r[1] * (k - 1) ** 4 + r[2] * (k - 1) ** 3 + r[3] * (k - 1) ** 2 + r[4] * (
                        k - 1) + r[5]
            temp3 = r[0] * (k + 1) ** 5 + r[1] * (k + 1) ** 4 + r[2] * (k + 1) ** 3 + r[3] * (k + 1) ** 2 + r[4] * (
                        k + 1) + r[5]
            if (temp1 <= temp2 and temp1 <= temp3):
                break
        each_grain_area_histogram_inflection_position[i, 0] = k
    [I1, labeled_image] =getNewBinImageAndLabeled(size1[0],size1[1],labeled_image,each_grain_area_histogram_inflection_position,I_gray,I1)
    labeled_image = label(I1, connectivity=1)
    props = regionprops(labeled_image)
    numObjects = np.max(labeled_image)
    return  [I1,labeled_image,I_gray,props,numObjects]

def getOneGrainArea(props,numObjects):
    each_grain_area = [props[i].area for i in range(numObjects)]
    sort_each_grain_area = sorted(each_grain_area)
    a1 = sort_each_grain_area[:]
    thr = 0.05;
    start0 = 0
    end0 = 0
    for i in range(numObjects - 2):
        if (a1[i] < 100):
            continue
        if ((a1[i + 1] - a1[i]) / a1[i] < thr) and ((a1[i + 2] - a1[i + 1]) / a1[i + 1] < thr) and (
                (a1[i + 3] - a1[i + 2]) / a1[i + 2] < thr) and ((a1[i + 4] - a1[i + 3]) / a1[i + 3] < thr):
            start0 = i
            break
    for i in range(start0 + 1, numObjects - 2):
        temp = (a1[i + 3] - a1[i]) / a1[i]
        if (temp > 0.2):
            end0 = i
            break
    # 确定最优单个谷粒面积
    min_error = 1000000
    one_grain_area = 0
    for i in range(start0, end0 + 1):
        total_error = 0
        total_number = 0
        for j in range(start0, end0 + 1):
            temp1 = int((sort_each_grain_area[j] / sort_each_grain_area[i]))  # 取整数部分
            temp3 = sort_each_grain_area[j] / sort_each_grain_area[i]
            if (temp1 == 0 or temp1 == 1):
                total_error = total_error + abs((1 - temp3))
                total_number = total_number + 1
        total_error = total_error / total_number
        if (total_error < min_error):
            min_error = total_error
            one_grain_area = sort_each_grain_area[i]
    #print(one_grain_area)
    return [one_grain_area,sort_each_grain_area]

@jit(parallel = True)
def getWhiteConnected(height,width,labeled_image,I_gray,white_threshold):
    white_connected = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if(labeled_image[i,j]==0 and I_gray[i,j]>white_threshold):
                white_connected[i, j] = 1
            else:
                white_connected[i, j] = 0
    return white_connected

def remove_max_objects(ar, max_size, connectivity=1, in_place=False):
    # Raising type error if not int or bool
    if in_place:
        out = ar
    else:
        out = ar.copy()

    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        print("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    too_max = component_sizes > max_size
    too_max_mask = too_max[ccs]
    out[too_max_mask] = 0
    return out

@jit()
def getBudNumber(white_numObjects,size1,white_labeled,labeled_image):
    bud_number = np.zeros((white_numObjects, 1))
    for i in range(1,size1[0]-1):
        for j in range(1, size1[1] - 1):
            temp1 = white_labeled[i, j]
            if(temp1==0):
                continue
            if(bud_number[temp1-1,0]==1):
                continue
            if(temp1>0 and (labeled_image[i-1,j]>0 or labeled_image[i+1,j]>0 or labeled_image[i,j-1]>0 or labeled_image[i,j+1]>0)):
                bud_number[temp1-1,0]=1
    return bud_number
def getNumOfGrains(picturePath):
    [I1, labeled_image, I_gray, props, numObjects]=getProcessedBinaryImage(picturePath)
    [one_grain_area, sort_each_grain_area]=getOneGrainArea(props, numObjects)
    size1 = I1.shape
    threshold1 = 0.5
    totalNum = 0
    for k in range(numObjects):
        temp1 = int((sort_each_grain_area[k] / one_grain_area))  # 取整数部分
        temp2 = (sort_each_grain_area[k] / one_grain_area) - temp1
        if (temp2 > threshold1):
            totalNum = totalNum + temp1 + 1
        else:
            totalNum = totalNum + temp1
    # ------------------统计萌发种子数目--------------------
    white_threshold = 160  # 定义白色的阈值，灰度大于160
    white_connected = getWhiteConnected(size1[0], size1[1],labeled_image,I_gray,white_threshold)
    # 删除较小和较大的胚芽
    min_bud_area = int(one_grain_area / 50)  # 确定最小胚芽面积
    max_bud_area = int(one_grain_area / 3)  # 确定最大胚芽面积
    white_connected = white_connected > 0
    white_connected = morphology.remove_small_objects(white_connected, min_bud_area, connectivity=1)
    white_connected = remove_max_objects(white_connected, max_bud_area)
    # 统计胚芽数目
    white_labeled = label(white_connected, connectivity=1)
    white_props = regionprops(white_labeled)
    white_numObjects = np.max(white_labeled)
    bud_num = getBudNumber(white_numObjects,size1,white_labeled,labeled_image)
    total_bud_num = sum(bud_num)[0]
    return totalNum,total_bud_num


# @jit(parallel = True)
# def getBinaryImage(height,width,I):
#     size1 = I.shape
#     I1 = np.zeros((size1[0], size1[1]))
#     rgbSumMatrix = np.zeros((height, width))
#     rgbSumMatrix = I[:, :, 0] + I[:, :, 1] + I[:, :, 2]
#     for i in range(height):
#         for j in range(width):
#             temp = rgbSumMatrix[i, j]
#             a = 0.3258962 * I[i,j,0] - 0.4992596 * I[i,j,1] + 0.1733409 * I[i,j,2] + 128;
#             b = 0.1218128 * I[i,j,0] + 0.3785610 * I[i,j,1] - 0.5003738 * I[i,j,2] + 128;
#             if ((temp > 300) and (100 < a < 136) and (140 < b < 195)):
#                 I1[i, j] = 1
#             else:
#                 I1[i, j] = 0
#     return I1

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if request.form.get('sub'):
            newFileName,fileName=api_upload()
            if(newFileName is not 'fail'):
                file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
                image_data = open(os.path.join(file_dir, newFileName), "rb").read()
                image_data=base64.b64encode(image_data).decode()
                totalNum, total_bud_num=getNumOfGrains(os.path.join(file_dir, newFileName))
                germination_rate=(total_bud_num*100)/totalNum
                germination_rate=str(round(germination_rate,2))+'%'
                #print(newFileName)
                # binImagePath=api_saveBinImage(os.path.join(file_dir, newFileName),newFileName)
                # bin_image_data=open(binImagePath,'rb').read()
                # bin_image_data=base64.b64encode(bin_image_data).decode()



            return render_template('index.html',image_data=image_data,totalNum=totalNum,total_bud_num=int(total_bud_num),germination_rate=germination_rate,fileName=fileName)
    image_data = open("default.jpg", "rb").read()
    image_data = base64.b64encode(image_data).decode()
    fileName="example.jpg"
    totalNum=74
    total_bud_num=48
    germination_rate=str(64.86)+'%'
    return render_template('index.html',totalNum=totalNum,total_bud_num=total_bud_num,germination_rate=germination_rate,image_data=image_data,fileName=fileName)

# 上传文件
#@app.route('/up_photo', methods=['POST'], strict_slashes=False)


if __name__ == '__main__':
    app.run()
