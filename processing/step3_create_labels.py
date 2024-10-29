from osgeo import gdal
import geopandas as gpd
from shapely.geometry import MultiPolygon,Polygon,shape,mapping,GeometryCollection
from shapely.ops import unary_union
import rasterio.features
import numpy as np
import rasterio as rio 
from rasterio.mask import mask
from datetime import datetime
import os 
import fiona
from geocube.api.core import make_geocube
from tqdm import tqdm
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None


def create_img(img,ar,output,dtype=gdal.GDT_Int16):
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(output,img.RasterXSize,img.RasterYSize,len(ar),dtype)
    for n,a in enumerate(ar,1):
        dst.GetRasterBand(n).WriteArray(a)
    dst.SetProjection(img.GetProjection())
    dst.SetGeoTransform(img.GetGeoTransform())
    dst.FlushCache()

def get_date_from_name(name):
    year = name[11:15]
    month = name[15:17]
    day = name[17:19]
    return f'{year}-{month}-{day}'

def get_bounds(im,projection=None,polygon=True):
    if projection is not None:
      src = gdal.Open(im)
      out = '/vsimem/temp.tif'
      gdal.Warp(out,src,dstSRS=projection)
      ds = gdal.Open(out)
    else:
      ds = gdal.Open(im)

    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    if polygon:
        return Polygon([[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]])
    else:
        return (xmin,ymin,xmax,ymax)

def create_poly_mask(file):
    data = rio.open(file)
    mask_ = data.read(1)
    mask_[mask_!=0]=1

    with rio.open(file) as src:
        shapes = list(rasterio.features.shapes(mask_, transform=src.transform))

    for i in shapes:
      if i[1]==1:
        poly = shape(i[0])
        poly = MultiPolygon([poly])
      
    return poly

def clip_img(img,polygon,output='',noData=0):


  with rio.open(img) as im:
    extent = mapping(polygon)
    imgRec,imgRec_affine= mask(im,
                               [extent],
                               nodata=noData,
                               crop=True)
    img_meta = im.meta.copy()
    img_meta.update({'transform': imgRec_affine,
                       'height': imgRec.shape[1],
                       'width': imgRec.shape[2],
                       'nodata': noData})

  if output!='':
    with rio.open(output, 'w', **img_meta) as ex:
      ex.write(imgRec)
  return imgRec


def clip_img_gdal(output,inp_file,prj,xRes,yRes,extent,nodata=None):
    outDs = gdal.Warp(output,
                      inp_file,
                      dstSRS=prj,
                      xRes=xRes,
                      yRes=yRes,
                      outputBounds=extent,
                      srcNodata = nodata,
                      resampleAlg='cubic',
                      format='GTiff')

    outDs = None

def create_labels_prodes(file1,file2,gdf,outpath,iterable=None,mask_file=None,verbose=False):
    if not os.path.exists(outpath):
      os.makedirs(outpath)
    output = os.path.join(outpath,os.path.basename(outpath)+'.tif')

    d1 = get_date_from_name(os.path.basename(file1))
    d2 = get_date_from_name(os.path.basename(file2))

    #TODO: Change dates here if needed
    # filtering based on the date range
    # d1 = '20201231'
    d2 = '20211231'
    if verbose:
      print(f"FILTERING BY DATES: {d1} AND {d2}")
    gdf = gdf[(gdf['image_date']>d1) & (gdf['image_date']<=d2)]

    poly1 = create_poly_mask(file1)
    poly2 = create_poly_mask(file2)
    
    # poly = unary_union([poly1,poly2])
    poly = poly1.intersection(poly2)
    area = poly.area

    if isinstance(poly,GeometryCollection):
      poly = poly.geoms[0]

    if verbose:
        print("polygons created")

    gdf2 = gpd.clip(gdf,poly)
    if verbose:
      print("Clip")

    if len(gdf2)>0:
      temp = os.path.splitext(output)[0] + "_temp.tif"
      
      gdf2['label'] = [1]*len(gdf2)
      out_grid= make_geocube(vector_data=gdf2, measurements=["label"], resolution=(-10, 10))
      out_grid["label"].rio.to_raster(temp)

      img = gdal.Open(file1)
      prj = img.GetProjection()
      xRes = yRes = img.GetGeoTransform()[1]
      extent = poly.bounds
      if verbose:
        print(extent)

      output_f1 = os.path.splitext(output)[0] + "_file1.tif"
      output_f2 = os.path.splitext(output)[0] + "_file2.tif"

      clip_img(file1,poly,output_f1)
      clip_img(file2,poly,output_f2)

      if verbose:
        print(os.path.exists(output_f1),output_f1)
        print(os.path.exists(output_f2),output_f2)

      extent = get_bounds(output_f1,polygon=False)
      outtemp = os.path.splitext(output)[0]+"_outtemp.tif"
      clip_img_gdal(outtemp,temp,prj,xRes,yRes,extent)

      os.remove(temp)
      if verbose:
        print(os.path.exists(temp),temp)

      file1 = file1.replace('_EVI','')
      file2 = file2.replace('_EVI','')
      file1 = file1.replace('_NDVI','')
      file2 = file2.replace('_NDVI','')
      if mask_file: 
        img = gdal.Open(outtemp)
        ar = img.ReadAsArray()

        mask_file = np.ones(ar.shape)
        for file_ in [file1,file2]:
          
          msk_file = os.path.splitext(file_)[0]+"_mask.tif"
          temp1 = os.path.splitext(msk_file)[0]+'_temp.tif'
          if verbose:
            print(temp1,msk_file)
            print(os.path.exists(temp1),os.path.exists(msk_file))
          gdal.Warp(temp1,msk_file,xRes=10, yRes=10, resampleAlg="near")#'cubic')
          
          mask_file1 = clip_img(temp1,poly)
          mask_file1 = mask_file1[0]
          mask_file[(mask_file1 == 3) | (mask_file1 == 7)| (mask_file1 == 8)| (mask_file1 == 9)| (mask_file1 == 10)] = 0
          os.remove(temp1)
        mask_file1 = None
        ar[mask_file==0] = 2
        ar = ar.astype(np.uint8)
        ar = cv2.medianBlur(ar,5)
        create_img(img,[ar],output,dtype=gdal.GDT_Byte)

        img = None
        os.remove(outtemp)

    else:
      if iterable:
        iterable.set_description(f"0 labels ({os.path.basename(file1)[33:44]}): {d1} and {d2}")

    return area,poly

def nearest(items, pivot, files):
    files = [i for i in files if datetime.strptime(os.path.basename(i)[11:19],"%Y%m%d")>pivot]
    items = [i for i in items if i>pivot]
    return [i for i in files if os.path.basename(i)[11:19]==min(items, key=lambda x: abs(x - pivot)).strftime("%Y%m%d")][0]
    


def find_dates(p2020,p2021,gdf):
    dates1 = [get_date_from_name(os.path.basename(i)) for i in p2020] 
    dates1.sort()
    d1 = dates1[-1]
    dates1 = [datetime.strptime(i,"%Y-%m-%d") for i in dates1]
    
    dates2 = [get_date_from_name(os.path.basename(i)) for i in p2021] 
    dates2.sort()
    d2 = dates2[-1]
    dates2 = [datetime.strptime(i,"%Y-%m-%d") for i in dates2]

    gdf = gdf[(gdf['image_date']>d1) & (gdf['image_date']<=d2)]
    if len(gdf)>0:
      selected_dates = []
      uniques = gdf['image_date'].unique()
      [(i,j) for i in uniques for j in uniques if i!=j  if i<j]
      for i in uniques:
        date = datetime.strptime(i,"%Y-%m-%d")
        near = nearest(dates2, date,p2021)
        selected_dates.append(near)
      return d1,selected_dates

def run(path,gdf,outpath,orbit_tile='',verbose=False):

    files2021 = [os.path.join(path,i) for i in os.listdir(path) if i[11:15]=='2021' and i.endswith('.tif') ]
    files2020 = [os.path.join(path,i) for i in os.listdir(path) if i[11:15]=='2020'and i.endswith('.tif')]

    if orbit_tile!="":
      files2021 = [i for i in files2021 if os.path.basename(i)[33:44]==orbit_tile]
      files2020 = [i for i in files2020 if os.path.basename(i)[33:44]==orbit_tile]

    with rio.open(files2020[0]) as src:
        crs = src.crs

    poly = get_bounds(files2020[0],projection=gdf.crs)
    gdf = gdf[gdf.intersects(poly)]
    gdf2 = gdf.to_crs(crs)
    try:
      d1,selected_dates = find_dates(files2020,files2021,gdf2)
    except TypeError:
      pass
    else:
      d1 = d1.replace('-','')

      selected_dates = list(set(selected_dates))

      file1 = [i for i in files2020 if os.path.basename(i)[11:19]==d1][0]

      for file2 in tqdm(selected_dates):
          n20 = os.path.basename(file1)
          n21 = os.path.basename(file2)
          if n20[33:44] == n21[33:44]:
            output = os.path.join(outpath,f'{n20[33:44]}-{n20[11:19]}_{n21[11:19]}.tif')
            if not os.path.exists(output):

              create_labels_prodes(file1,file2,gdf2,output,verbose=verbose)

def run_files(file1,file2,gdf,outpath,mask_files=True,verbose=False):
  with rio.open(file1) as src:
        crs = src.crs

  poly = get_bounds(file1,projection=gdf.crs)
  gdf = gdf[gdf.intersects(poly)]
  gdf.to_crs(crs,inplace=True)
  
  area,poly = create_labels_prodes(file1,file2,gdf,outpath,mask_file=mask_files,verbose=verbose)
  return area,poly




def run_all(path,shp,outpath):
  gdf = gpd.read_file(shp)

  for i in tqdm(os.listdir(path)):
  # for i in os.listdir(path):
    p = os.path.join(path,i)
    p2020 = os.path.join(p,'2020')
    p2021 = os.path.join(p,'2021')
    outp = os.path.join(outpath,i)
    if not os.path.exists(outp):
      os.mkdir(outp)
    run(p2020,p2021,gdf,outp)

if __name__=="__main__":
  import argparse
  parser =  argparse.ArgumentParser(description="Create labels")
  parser.add_argument("yearly_deforestation",type=str,help="Yearly deforestation file, from TerraBrasilis website.")
  parser.add_argument("file1",type=str,help="File 1, from year 2020")
  parser.add_argument("file2",type=str,help="File 2, from year 2021")
  parser.add_argument("outpath",type=str,help="Path to store the files")
  parser.add_argument('-v','--verbose',type=bool,help="If you need to print everything that is happening, set verbose as True",default=False)
  args = parser.parse_args()
  shp = args.yearly_deforestation
  outpath = args.outpath
  file1 = args.file1
  file2 = args.file2
  verbose = args.verbose

  tile = os.path.basename(file1)[33:44]
  name1 = os.path.basename(file1)[11:19]
  name2 = os.path.basename(file2)[11:19]

  outpath = os.path.join(outpath,f'{tile}_{name1}_{name2}')
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  gdf = gpd.read_file(shp)
  area,poly = run_files(file1,file2,gdf,outpath,mask_files=True,verbose=verbose)

  file = os.path.join(outpath,'calc_area.csv')
  name = f'{os.path.basename(file1)}_{os.path.basename(file2)}'

  if not os.path.exists(file):  
    df = pd.DataFrame({'area(km2)':[area/1_000_000],'name':[name]})
  else:
    df = pd.read_csv(file)
    df = df[['area(km2)','name']]
    row = [area/1_000_000,name]
    if name not in df['name']:
      df.loc[len(df.index)] = row 
  
  df.to_csv(file)

  if verbose:
    print("Done step 3 :)")


  