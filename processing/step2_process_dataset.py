from osgeo import gdal
import os
from zipfile import ZipFile
import re
from tqdm import tqdm
import argparse

def get_bands(file,bands,pix_size,pattern=''):
    tile = os.path.basename(file)[38:44]
    pix_size=f'_{pix_size}m'
    if pattern=='':
        pattern = f'.*({tile}).*B({"|".join(bands)}){pix_size}.jp2$'
    bands_to_get = re.compile(pattern)
    return bands_to_get

def read_zip(file,bands_to_get):
    with ZipFile(file,'r') as zipfile:
        bands = [x for x in zipfile.namelist() if re.match(bands_to_get,x)]
        bands.sort()
        ar = []
        for b in bands:
            str_bands = '/vsizip/%s/%s'%(file,b)
            ds = gdal.Open(str_bands)
            ar.append(ds.ReadAsArray())
    return ar,ds

def read_bands(file,bands,pix_size,pattern = ''):
    bands_to_get = get_bands(file,bands,pix_size,pattern=pattern)
    ar,ds = read_zip(file,bands_to_get)
    return ar,ds
    

def create_img(img,ar,output,dtype=gdal.GDT_Int16):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(ar),dtype)
    for n,a in enumerate(ar,1):
        dst.GetRasterBand(n).WriteArray(a)
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()

def create_compos(path,outpath,calc_EVI = False,calc_NDVI = False,create_mask=False):
    for i in tqdm(os.listdir(path)):
        ar = None
        if i.endswith('.zip'):
            outpath_ = os.path.join(outpath,i[33:44])
            if not os.path.exists(outpath_):
                os.makedirs(outpath_)
                print(f"Files will be saved in {outpath_}")
            file = os.path.join(path,i)
            # output = os.path.join(outpath,os.path.basename(os.path.splitext(file)[0])+'.tif')
            output = os.path.join(outpath_,os.path.basename(os.path.splitext(file)[0])+'.tif')
            bands = ['02','03','04','08']
            pix_size = 10
            
            if not os.path.exists(output):
                ar,img = read_bands(file,bands,pix_size)
                create_img(img,ar,output,dtype=gdal.GDT_Int16)

            if create_mask:
                output_mask = os.path.splitext(output)[0]+"_mask.tif"
                if not os.path.exists(output_mask):
                    tile = os.path.basename(file)[38:44]
                    pattern = f'.*({tile}).*SCL_({20})m.jp2$'                    
                    ar,img = read_bands(file,bands,20,pattern=pattern)
                    create_img(img,ar,output_mask,dtype=gdal.GDT_Int16)

            if calc_EVI:            
                outputEVI = os.path.join(outpath_,os.path.basename(os.path.splitext(file)[0])+'_EVI.tif')
                # if not os.path.exists(outputEVI):
                if ar is None:
                    ar,img = read_bands(file,bands,pix_size)
                B08 = ar[3]
                B04 = ar[2]
                # B03 = ar[1]
                B02 = ar[0]
                # C1 = 6
                # C2 = 7.5
                # L=1
                # G = 2.5
                # EVI = G*((B08-B04)/(B08+(C1*B04)-(C2*B02)+L))
                EVI = 2.5 * ((B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1))
                create_img(img,[EVI],outputEVI,dtype=gdal.GDT_Float32)
                
                # outputEVI = os.path.join(outpath_,os.path.basename(os.path.splitext(file)[0])+'_EVI2.tif')
                # EVI2 = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
                # create_img(img,[EVI2],outputEVI,dtype=gdal.GDT_Float32)
            
            if calc_NDVI:
                outputNDVI = os.path.join(outpath_,os.path.basename(os.path.splitext(file)[0])+'_NDVI.tif')
                if not os.path.exists(outputNDVI):
                    if ar is None:
                        ar,img = read_bands(file,bands,pix_size)    
                    B08 = ar[-1]
                    B04 = ar[-2]
                    NDVI = (B08-B04)/(B08+B04)
                    create_img(img,[NDVI],outputNDVI,dtype=gdal.GDT_Float32)

if __name__=="__main__":
    parser =  argparse.ArgumentParser(description="Creating compositions")
    parser.add_argument("path",type=str,help="Path where all the files were downloaded.")
    parser.add_argument("outpath",type=str,help="Path to store the output files.")
    args = parser.parse_args()
    
    path = args.path
    outpath = args.outpath
    create_mask = True
    EVI_ = True
    NDVI_= True

    for p in os.listdir(path):
        path_ = os.path.join(path,p)
        if p.startswith('T') and os.path.isdir(path_):
            create_compos(path_,outpath,create_mask=create_mask,calc_EVI=EVI_,calc_NDVI=NDVI_)
