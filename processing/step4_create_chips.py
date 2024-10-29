from osgeo import gdal
import os
from tqdm import tqdm
import numpy as np
# import sys
import matplotlib.pyplot as plt
import argparse
import cv2



def clip_img(output, inp_file, prj, xRes, yRes, extent, nodata=None):
    outDs = gdal.Warp(output,
                      inp_file,
                      dstSRS=prj,
                      xRes=xRes,
                      yRes=yRes,
                      outputBounds=extent,
                      srcNodata=nodata,
                      resampleAlg='near',
                      format='GTiff')

    outDs = None

def norm(array,pct=-1):
    if pct!=-1:
        vals = list(array.reshape(-1))
        vals.sort()
        maxv = vals[int(len(vals)*pct)]
        minv = vals[int(len(vals)*(1-pct))]
    else:
        minv = np.min(array)
        maxv = np.max(array)

    array = ((array-minv)/(maxv-minv))
    array[array>1] = 1
    array[array<0] = 0
    return array

def save_tif(img,ar,output,dtype=gdal.GDT_Byte):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(ar),dtype)
    for n,a in enumerate(ar,1):
        dst.GetRasterBand(n).WriteArray(a)
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()

def norm_and_save(file,
                  output,
                  format,
                  pct = -1,
                  pct_class1=0,
                  rotations=[],
                  label=False,
                  bands=[0,1,3]
                  ):

    img = gdal.Open(file)
    arr = img.ReadAsArray()

    if img.RasterCount==1:
        if label:
            un = np.unique(arr)
            pct_clouds = len(arr[arr==2])/len(arr.reshape(-1))
            if (0 not in un or 1 not in un) or pct_clouds>0.1:
                return
            arr[arr==2]=0
            
            ########################### To add only 20% ###########################
            if pct_class1>0:
                pct_0_and_1 = len(arr[arr==1])/len(arr.reshape(-1))
                if pct_0_and_1<pct_class1:#0.2:
                    return
            #######################################################################
        else:
            total = len(arr.reshape(-1))    
            zeros = len(arr[arr==0])        
            pct_zero = (zeros/total)*100    

            if pct_zero>0:
                return
            
            arr = norm(arr,pct=pct)

        if format=='tif':
            save_tif(img,[arr],output)
            rotate_aug(rotations,output,arr,img=img)
        else:
            try:
                if not label:
                    arr*=255
                    arr = arr.astype(np.uint8)
                cv2.imwrite(output,arr)
                rotate_aug(rotations,output,arr,img=None)
            except cv2.error as e:
                raise Exception(f'Error with the format you choose: "{e}"')

    else:
        total = len(arr.reshape(-1))    
        zeros = len(arr[arr==0])        
        pct_zero = (zeros/total)*100    

        if pct_zero>0:
            return
        arrays = []
        for i in range(img.RasterCount):

            ar = norm(arr[i],pct=pct)
            arrays.append(ar)

        arr = np.array(arrays)

        if format=='tif':
            arr = np.array(arrays)
            save_tif(img,arr,output,dtype=gdal.GDT_Float32)
            rotate_aug(rotations,output,arr,img=img)
        else:
            arr = np.array(arrays)
            arr*=255
            arr = arr.astype(np.uint8)
            # arr = cv2.merge((arr[0],arr[1],arr[3]))
            arr = cv2.merge((arr[bands[0]],arr[bands[1]],arr[bands[2]]))
            cv2.imwrite(output,arr)
            rotate_aug(rotations,output,arr,img=None)

def rotate_aug(rotations,output,arr,img=None):
    for angle in rotations:
        output_angle = os.path.splitext(output)[0]+f"_rot={angle}{os.path.splitext(output)[-1]}"
        
        if img is None:
            ar = cv2.split(arr)
            ar = rotate(np.array(ar),angle)
            if len(ar.shape)==3:
                ar = cv2.merge(ar)
            cv2.imwrite(output_angle,ar)
        else:
            ar = rotate(arr,angle)
            if len(ar.shape)==2:
                ar = [ar]
            save_tif(img,ar,output_angle,dtype=gdal.GDT_Float32)


def rotate(img,angle):
    if len(img.shape)==2:
        img = [img]
    rotated_image = []
    for image in img:

        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rt = cv2.warpAffine(
            src=image, M=rotate_matrix, dsize=(width, height))
        rotated_image.append(rt)

    if len(rotated_image)==1:
        return rotated_image[0]
    else:
        return np.array(rotated_image,dtype=rotated_image[0].dtype)
    

def get_bounds(ds):
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return xmin, ymin, xmax, ymax

def create_polygons(img,xsize,ysize,px_size,shiftx = 0,shifty = 0):
    xsize *= px_size
    ysize *= px_size

    xmin, ymin, xmax, ymax = get_bounds(img)

    xmin += shiftx
    ymin += shifty
    xmax += shiftx
    ymax += shifty

    yrg = np.arange(ymax,ymin,-ysize/1.5)
    xrg = np.arange(xmin,xmax,xsize/1.5)

    polygons = []
    for i in yrg:
        for j in xrg:
            y = i + ysize
            x = j + xsize  

            polygons.append([j, i, x, y])
    return polygons

dic_bands = {'B':0,'G':1,'R':2,'N':3}

def run(input_file1,
        input_file2,
        label_file,
        size,
        i_outpath_1,
        i_outpath_2,
        l_outpath,
        pct,
        shift=False,
        angles=[90,180],
        norm_pct=-1,
        bands = 'BGN'
        ):
    
    bands = [dic_bands[i] for i in bands]
    
    img_inp_1 = gdal.Open(input_file1)
    img_inp_2 = gdal.Open(input_file2)
    img_lab = gdal.Open(label_file)
    res = img_inp_1.GetGeoTransform()[1]
    proj = img_inp_1.GetProjection()
    polygons = create_polygons(img_inp_1, size, size, res)
    if shift:
        shiftx = size*(img_inp_1.GetGeoTransform()[1]/4)
        shifty = size*(img_inp_1.GetGeoTransform()[1]/4)
        print(f"Changing sizes to x+{shiftx} and y+{shifty}")
        polygons += create_polygons(img_inp_1, size, size, res,shiftx = shiftx,shifty = shifty)
        shiftx = size*(img_inp_1.GetGeoTransform()[1]/2)
        shifty = size*(img_inp_1.GetGeoTransform()[1]/2)
        print(f"Changing sizes to x+{shiftx} and y+{shifty}")
        polygons += create_polygons(img_inp_1, size, size, res,shiftx = shifty,shifty = shifty)
    
    for n, p in enumerate(tqdm(polygons), 1):
        name = os.path.splitext(os.path.basename(input_file1))[0] + f'_{str(n).zfill(4)}.{outformat}'
        temp = '/vsimem/temp.tif'
        output = os.path.join(i_outpath_1, name)

        files = []
        files.append(output)

        if not os.path.exists(output):
            clip_img(temp, img_inp_1, proj, res, res, p)
            norm_and_save(temp,
                          output,
                          format=outformat,
                          rotations=angles,
                          pct=norm_pct,
                          bands=bands#[0,1,3]
                          )

        output = os.path.join(i_outpath_2, name)
        files.append(output)
        if not os.path.exists(output):
            clip_img(temp, img_inp_2, proj, res, res, p)
            norm_and_save(temp,
                          output,
                          format=outformat,
                          rotations=angles,
                          pct=norm_pct,
                          bands=bands#[0,1,3]
                          )

        output = os.path.join(l_outpath, name)
        files.append(output)
        if not os.path.exists(output):
            clip_img(temp, img_lab, proj, res, res, p)
            norm_and_save(temp,
                          output,
                          format=outformat,
                          pct_class1=pct,
                          rotations=angles,
                          label=True
                          )

        files_exists = [True if os.path.exists(i) else False for i in files]
        files_rotated = [os.path.splitext(i)[0]+f"_rot={angle}{os.path.splitext(i)[-1]}" for i in files for angle in angles]
        if not all(files_exists):
            for f in files+files_rotated:
                if os.path.exists(f):
                    os.remove(f)


if __name__=="__main__":
    parser =  argparse.ArgumentParser(description="Create Chips")
    # parser.add_argument("input_file1",type=str,help="Input File 1")
    # parser.add_argument("input_file2",type=str,help="Input File 2")
    # parser.add_argument("label_file",type=str,help="Label file (output step 3)")
    parser.add_argument("path",type=str,help="path where the labels are stored")
    parser.add_argument("outpath",type=str,help="Path where all the files will be downloaded")
    parser.add_argument('-s',"--size",type=int,help="Size of the chips",default=256)
    parser.add_argument('-f','--format',type=str,help="output data format. Allowed: 'tif' or 'png' ",default='png')
    parser.add_argument('-n','--normalize',type=float,help='Percentage of maximum value to be used (e.g. 98%% highest value',default=-1)
    parser.add_argument('-v','--verbose',type=bool,help="If you need to print everything that is happening, set verbose as True",default=False)
    parser.add_argument('-p','--pct',type=float,help='Minimum percentage of class 1.',default=0.1)
    parser.add_argument('--angles', nargs='+', type=int,default=[90,180])


    args = parser.parse_args()
    # input_file1 = args.input_file1
    # input_file2 = args.input_file2
    # label_file = args.label_file
    path = args.path
    input_file1 = os.path.join(path,os.path.basename(path)+'_file1.tif')
    input_file2 = os.path.join(path,os.path.basename(path)+'_file2.tif')
    label_file = os.path.join(path,os.path.basename(path)+'.tif')

    verbose = args.verbose
    outputpath = args.outpath
    size = args.size
    outformat = args.format
    pct = args.pct
    angles = args.angles
    norm_pct = args.normalize

    input_file1 = os.path.join(path,os.path.basename(path)+'_file1.tif')
    input_file2 = os.path.join(path,os.path.basename(path)+'_file2.tif')
    label_file = os.path.join(path,os.path.basename(path)+'.tif')

    l_outpath = os.path.join(outputpath, str(size), 'labels')
    if not os.path.exists(l_outpath):
        os.makedirs(l_outpath)

    i_outpath_1 = os.path.join(outputpath, str(size), 'inputs1')
    if not os.path.exists(i_outpath_1):
        os.makedirs(i_outpath_1)

    i_outpath_2 = os.path.join(outputpath, str(size), 'inputs2')
    if not os.path.exists(i_outpath_2):
        os.makedirs(i_outpath_2)

    run(input_file1,input_file2,label_file,size,i_outpath_1,i_outpath_2,l_outpath,pct,angles=angles,norm_pct=norm_pct)
    print('Done step 4 :)')



