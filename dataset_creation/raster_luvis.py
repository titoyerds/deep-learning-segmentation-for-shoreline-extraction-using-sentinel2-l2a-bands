import processing

for i in range(31, 32):
    buffer_layer = QgsProject.instance().mapLayersByName(f'Buffered{i}')
    bands_layer = QgsProject.instance().mapLayersByName(f'luvis_shoreline_grid_{i}_Bands')
    
    if not buffer_layer or not bands_layer:
        print(f'Skipping {i} — layer not found')
        continue
    
    ext = bands_layer[0].extent()
    crs = bands_layer[0].crs().authid()
    extent_str = f'{ext.xMinimum()},{ext.xMaximum()},{ext.yMinimum()},{ext.yMaximum()} [{crs}]'
    
    processing.run("gdal:rasterize", {
        'INPUT': buffer_layer[0],
        'BURN': 1,
        'UNITS': 0,
        'WIDTH': 4096,
        'HEIGHT': 4096,
        'EXTENT': extent_str,
        'NODATA': -1,       # -1 so it doesn't conflict with 0 or 1
        'DATA_TYPE': 0,     # Byte
        'INIT': 0,          # pre-fill with 0
        'OUTPUT': f'C:/Users/Andrey Fritz/Downloads/dataset_creation/shoreline_grids/luzon_visayas/masks/luvis_shoreline_mask_{i}.tif'
    })
    
    print(f'Done: luvis_shoreline_mask_{i}.tif')