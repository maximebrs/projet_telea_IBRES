import numpy as np
from osgeo import gdal, ogr

def rasterize_samples(image_ref_path, vector_path, field_name='strate'):
    """
    Rastérise un shapefile sur la grille d'une image de référence.
    Retourne un array numpy où chaque pixel a la valeur du champ demandé.
    """
    # 1. Ouvrir l'image de référence pour récupérer dimensions et géoréférencement
    ds_ref = gdal.Open(image_ref_path)
    if ds_ref is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {image_ref_path}")
        
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize
    geo_transform = ds_ref.GetGeoTransform()
    projection = ds_ref.GetProjection()
    
    # 2. Créer un raster temporaire en mémoire
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    # 3. Ouvrir le vecteur
    shp_ds = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if shp_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {vector_path}")
    layer = shp_ds.GetLayer()
    
    # 4. Rastériser les polygones sur le raster vide
    # Initialisation à 0
    band = target_ds.GetRasterBand(1)
    band.Fill(0)
    
    # Ecrire la valeur du champ 'strate'
    gdal.RasterizeLayer(target_ds, [1], layer, options=[f"ATTRIBUTE={field_name}"])
    
    # 5. Récupérer le tableau numpy
    array = band.ReadAsArray()
    
    # Nettoyage
    ds_ref = None
    target_ds = None
    shp_ds = None
    
    return array