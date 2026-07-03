import ee
import datetime

tasks = []
nextGridID = 102
coordinates = [
    [121.47383160693376, 5.724871701876767, 121.83423441188138, 6.087498108742966]
]

# --- Project ID ---
GEE_PROJECT_ID = 'smart-exchange-474519-v0' 

# Authenticate the user token (Optional - usually only needed once) 
# ee.Authenticate() 

# Initialize GEE, explicitly linking to the required Google Cloud Project
try:
    ee.Initialize(project=GEE_PROJECT_ID)
    print("GEE initialized successfully. Proceeding with data query...")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize GEE. Check your Project ID or run ee.Authenticate() again. Error: {e}")
    exit()


# ── s2cloudless masking (pre-join approach — crash-safe) ─────────────────────
def apply_cloud_mask_from_joined(image):
    """
    Called after the S2 SR and S2_CLOUD_PROBABILITY collections are
    pre-joined. Each image already carries its cloud_prob band as a property.
    Falls back to unmasked image if no match found (safe fallback).
    """
    # Get the matched cloud probability image (may be null if no match)
    cloud_prob_image = ee.Image(image.get('cloud_prob'))
 
    # Safe fallback: if no cloud prob image found, return image unmasked
    # (it will still be filtered by CLOUDY_PIXEL_PERCENTAGE at collection level)
    cloud_prob = cloud_prob_image.select('probability')
    is_cloud   = cloud_prob.gt(50)  # mask pixels with >50% cloud probability
 
    # Dilate by 100m to catch cloud edges and shadows
    cloud_dilated = is_cloud.focal_max(
        radius     = 100,
        units      = 'meters',
        kernelType = 'circle'
    )
 
    return image.updateMask(cloud_dilated.Not())


for i in range(len(coordinates)):
    # --- 1. Define Study Parameters ---
    # Use a specific region, defined as a bounding box [min_lon, min_lat, max_lon, max_lat]
    AOI_COORDINATES  = coordinates[i]
    REGION_NAME      = f'min_shoreline_grid_{nextGridID}'
    nextGridID      += 1

    # Date filtering (Look for cloud-free images within this period)
    START_DATE       = '2023-01-01'
    END_DATE         = '2026-06-11'

    # Maximum cloud cover percentage allowed in the image (0-100)
    CLOUD_FILTER_MAX = 30

    # Target spatial resolution for the export (10m)
    TARGET_SCALE     = 10 

    # Select all available spectral bands (B1 through B12, and B8A)
    ALL_SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    # --- 2. Define Area of Interest (AOI) ---
    AOI = ee.Geometry.Rectangle(AOI_COORDINATES)

    # --- 3. Load s2cloudless collection (same AOI + date range) ---
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                          .filterBounds(AOI)
                          .filterDate(START_DATE, END_DATE))
 
    # --- 4. Load and filter Sentinel-2 SR collection ---
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(AOI)
                       .filterDate(START_DATE, END_DATE)
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER_MAX)))
    
    # Check collection size
    count = s2_collection.size().getInfo()
    print(f"\nImages found for {REGION_NAME}: {count}")

    if count == 0:
        print(f"No images found for {REGION_NAME}, skipping...")
        continue

    # --- 5. Apply s2cloudless pixel-level cloud masking to every image ---
    # --- Pre-join: attach cloud_prob to each SR image BEFORE mapping ─────────
    # This avoids the null crash — images without a match are simply excluded
    join_condition = ee.Filter.equals(
        leftField  = 'system:index',
        rightField = 'system:index'
    )
 
    joined_col = ee.Join.saveFirst('cloud_prob').apply(
        primary   = s2_collection,
        secondary = s2_cloudless_col,
        condition = join_condition
    )
 
    # Now safely map — every image in joined_col has a cloud_prob property
    masked_collection = ee.ImageCollection(joined_col).map(apply_cloud_mask_from_joined)
 
    # --- 6. Build median composite from all cloud-masked images ---
    # Median naturally fills remaining gaps where clouds were removed
    median_composite = masked_collection.select(ALL_SPECTRAL_BANDS).median().clip(AOI)
 
    # --- 7. Log reference image info (least cloudy single image) ---
    try:
        best_single      = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        timestamp_ms     = best_single.get('system:time_start').getInfo() # Convert milliseconds to datetime object
        acquisition_date = datetime.datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
        print(f"Reference image date       : {acquisition_date}")
        print(f"Reference image cloud cover: {best_single.get('CLOUDY_PIXEL_PERCENTAGE').getInfo():.2f}%")
    except Exception as e:
        print(f"Could not retrieve reference image info: {e}")

    # --- 8. Export to Google Drive ---
    task = ee.batch.Export.image.toDrive(
        image          = median_composite,
        description    = f'{REGION_NAME}_Sentinel2_Bands',
        folder         ='GEE_Thesis_Mindanao',         # Folder name in your Google Drive
        fileNamePrefix = f'{REGION_NAME}_Bands',
        scale          = TARGET_SCALE,                 # Defines the output resolution (10m)
        region         = AOI.getInfo()['coordinates'], # Defines the geographic extent
        fileFormat     = 'GeoTIFF',                    # Essential for retaining spectral data
        maxPixels      = 1e10                          # Increase if the region is very large
    )

    # Start the task
    task.start()
    tasks.append((REGION_NAME, task))

    print(f"Export task started for {REGION_NAME}.")
    print(f"Task status: {task.status()['state']}")

print()
for name, t in tasks:
    print(f"{name}: {t.status()['state']}")